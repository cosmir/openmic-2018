#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''OpenMIC train-test partitioning'''

import argparse
import sys

from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def process_args(args):
    '''Parse arguments from the command line'''

    parser = argparse.ArgumentParser(description='Split OpenMIC-2018 data into train and test')

    parser.add_argument('metadata',
                        help='Path to metadata.csv',
                        type=str)

    parser.add_argument('labels',
                        help='Path to sparse-labels.csv',
                        type=str)

    parser.add_argument('-s', '--seed', dest='seed',
                        default=20180821,
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

    return vars(parser.parse_args(args))


def load_label_matrix(metadata_file, label_file):
    '''Load metadata and sparse labels from CSV

    Returns
    -------
    sample_keys : pd.Series
        Ordered array matching row numbers to sample keys

    label_matrix : pd.DataFrame
        Sparse (nan-populated) array matching row numbers to instrument relevance scores
    '''
    # Load in the data
    meta = pd.read_csv(metadata_file)
    labels = pd.read_csv(label_file)

    # Get a row index on sample keys
    skey = meta[['sample_key']].reset_index()

    # Join the tables to get row index -> (label, relevance)
    skm = pd.merge(skey, labels, how='inner')

    # Pivot the table to get a row-major annotation vector
    label_matrix = skm.pivot_table(columns='instrument',
                                   values='relevance',
                                   index='index')

    # And fill in an extra column for all-negative examples
    label_matrix['_negative'] = (label_matrix.max(axis=1) < 0) * 1.0

    return skey['sample_key'], label_matrix


def make_partitions(metadata, labels, seed, num_splits, ratio):
    '''Partition the open-mic data into train-test splits.

    The partitioning logic is as follows:

        1. Match each track with its most positive label association
            1a. if no positive associations are found, label it as '_negative'
        2. Use sklearn StratifiedShuffleSplit to make balanced train-test partitions
        3. Save each partition as two index csv files
    '''
    sample_keys, label_matrix = load_label_matrix(metadata, labels)

    splitter = StratifiedShuffleSplit(n_splits=num_splits,
                                      random_state=seed,
                                      test_size=1-ratio)

    # Convert sparse multi-labels to multiclass label array
    labels = label_matrix.idxmax(axis=1)

    # Loop over folds; we can use the label vector as if it was features here
    # since we never actually look at the "features"
    for fold, (train_idx, test_idx) in tqdm(enumerate(splitter.split(labels, labels))):
        train_ser = sample_keys[train_idx].sort_values()
        test_ser = sample_keys[test_idx].sort_values()
        train_ser.to_csv('split{:02d}_train.csv'.format(fold), index=False)
        test_ser.to_csv('split{:02d}_test.csv'.format(fold), index=False)


if __name__ == '__main__':

    params = process_args(sys.argv[1:])
    make_partitions(**params)
