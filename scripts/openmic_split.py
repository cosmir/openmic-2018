#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''OpenMIC train-test partitioning'''

import argparse
import sys

from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def process_args(args):
    '''Parse arguments from the command line

    Parameters
    ----------
    args : list of str
        Command-line arguments, i.e., sys.argv[1:]

    Returns
    -------
    args_parsed : dict
        Dictionary of parsed arguments
    '''

    parser = argparse.ArgumentParser(description='Split OpenMIC-2018 data into train and test')

    parser.add_argument('metadata',
                        help='Path to metadata.csv',
                        type=str)

    parser.add_argument('labels',
                        help='Path to sparse-labels.csv',
                        type=str)

    parser.add_argument('--dupes', dest='dupe_file', type=str,
                        help='Path to track de-duplication index')

    parser.add_argument('-s', '--seed', dest='seed',
                        default=20180903,
                        help='Random seed',
                        type=int)

    parser.add_argument('-n', '--num-splits', dest='num_splits',
                        default=1,
                        help='Number of splits to generate',
                        type=int)

    parser.add_argument('-r', '--split-ratio', dest='ratio',
                        default=0.75,
                        help='Fraction of data for training',
                        type=float)

    parser.add_argument('-p', '--probability-ratio', dest='prob_ratio',
                        default=0.875, type=float,
                        help='Max/min allowable deviation of p(Y | train) / p(Y)')

    return vars(parser.parse_args(args))


def load_label_matrix(metadata_file, label_file, dupe_file=None):
    '''Load metadata and sparse labels from CSV

    Parameters
    ----------
    metadata_file : str
    label_file : str
        Paths to CSV files storing the openmic metadata and sparse label assignments

    dupe_file : str
        Path to CSV file storing a de-duplication mapping of sample keys to artist ids

    Returns
    -------
    sample_keys : pd.DataFrame
        Ordered array matching row numbers to sample keys and artist ids

    artist_labels : pd.DataFrame
        Sparse (nan-populated) array matching artists to instrument relevance scores

    label_matrix : pd.DataFrame
        Sparse (nan-populated array matching sample keys to instrument relevance scores
    '''
    # Load in the data
    meta = pd.read_csv(metadata_file)
    labels = pd.read_csv(label_file)

    if dupe_file:
        # Override the original artist id with the dedupe index
        # Store the original artist id as artist_id_orig
        dedupe = pd.read_csv(dupe_file)
        meta = meta.merge(dedupe, on='sample_key', suffixes=('_orig', ''))

    # Get a row index on sample keys
    skey = meta[['sample_key', 'artist_id']].reset_index()

    # Join the tables to get row index -> (label, relevance)
    skm = pd.merge(skey, labels, how='inner')

    # Pivot the table to get a row-major annotation vector
    label_matrix = skm.pivot_table(columns='instrument',
                                   values='relevance',
                                   index='index')

    artist_labels = pd.merge(label_matrix, skm[['artist_id']],
                             left_index=True,
                             right_index=True,
                             how='right').groupby('artist_id').mean()

    # And fill in an extra column for all-negative examples
    artist_labels['_negative'] = (artist_labels.max(axis=1) < 0) * 1.0

    label_matrix.index = meta['sample_key']

    return skey, artist_labels, label_matrix


def check_prob(label_matrix, idx, prob_ratio):
    '''Check that the probabilities in a sub-sample
    are within a tolerance of the full population.

    Parameters
    ----------
    label_matrix : pd.DataFrame
        Array of label assignments

    idx : iterable
        Indices of the target sub-sample

    prob_ratio:
        The target probability ratio

    Returns
    -------
    check_passed : bool
        True if the sub-sampled distribution is within tolerance
        False otherwise
    '''
    min_prob, max_prob = sorted([prob_ratio, 1./prob_ratio])

    all_dist_p = (label_matrix > 0).sum() / label_matrix.count()
    all_dist_n = (label_matrix <= 0).sum() / label_matrix.count()

    sub_dist_p = (label_matrix.loc[idx] > 0).sum() / label_matrix.loc[idx].count()
    sub_dist_n = (label_matrix.loc[idx] <= 0).sum() / label_matrix.loc[idx].count()

    # Make sure that for each class Y, we have the following conditions:
    #   P[Y = 1 | x in sample] >= P[Y = 1] * min_prob   # positive examples are not too unlikely
    #   P[Y = 1 | x in sample] <= P[Y = 1] * max_prob   # or too likely
    #   P[Y = 0 | x in sample] >= P[Y = 0] * min_prob   # likewise for negative examples
    #   P[Y = 0 | x in sample] <= P[Y = 0] * max_prob
    return (np.all(min_prob * all_dist_p.values <= sub_dist_p.values) and
            np.all(sub_dist_p.values <= max_prob * all_dist_p.values) and
            np.all(min_prob * all_dist_n.values <= sub_dist_n.values) and
            np.all(sub_dist_n.values <= max_prob * all_dist_n.values))


def make_partitions(metadata, labels, seed, num_splits, ratio, prob_ratio, dupe_file=None):
    '''Partition the open-mic data into train-test splits.

    The partitioning logic is as follows:

        1. Match each track with its most positive label association
            1a. if no positive associations are found, label it as '_negative'
        2. Use sklearn StratifiedShuffleSplit to make balanced train-test partitions
        3. Save each partition as two index csv files

    Parameters
    ----------
    metadata : str
        Path to metadata CSV file

    labels : str
        Path to sparse labels CSV file

    seed : None, np.random.RandomState, or int
        Random seed

    num_splits : int > 0
        Number of splits to generate

    ratio : float in [0, 1]
        Fraction of data to separate for training

    prob_ratio : float in [0, 1]
        Minimum probability ratio for P(Y | train) (or P(Y | test)) to P(Y)
    '''

    sample_keys, artist_labels, label_matrix = load_label_matrix(metadata, labels, dupe_file)

    splitter = StratifiedShuffleSplit(n_splits=num_splits * 1000,
                                      random_state=seed,
                                      test_size=1-ratio)

    # Convert sparse multi-labels to multiclass label array
    labels = artist_labels.idxmax(axis=1)

    # Loop over folds; we can use the label vector as if it was features here
    # since we never actually look at the "features"
    fold = 0
    for artist_train_idx, artist_test_idx in tqdm(splitter.split(labels, labels)):
        train_artists = artist_labels.index[artist_train_idx]
        test_artists = artist_labels.index[artist_test_idx]

        train_idx = sample_keys[sample_keys['artist_id'].isin(train_artists)]['sample_key'].sort_values()
        test_idx = sample_keys[sample_keys['artist_id'].isin(test_artists)]['sample_key'].sort_values()

        # Throw an exception if the train and test indices have common elements
        if set(train_idx) & set(test_idx):
            raise RuntimeError('Train and test indices overlap!')

        # Here's where we check for label deviation
        if (check_prob(label_matrix, train_idx, prob_ratio) and
            check_prob(label_matrix, test_idx, prob_ratio)):

            fold += 1
            train_idx.to_csv('split{:02d}_train.csv'.format(fold), index=False)
            test_idx.to_csv('split{:02d}_test.csv'.format(fold), index=False)

        if fold >= num_splits:
            break

    if fold < num_splits:
        raise ValueError('Unable to find sufficient splits. Try lowering the probability ratio tolerance.')


if __name__ == '__main__':

    params = process_args(sys.argv[1:])
    make_partitions(**params)
