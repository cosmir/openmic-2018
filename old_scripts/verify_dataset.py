#!/usr/bin/env python
# coding: utf8
'''Compute various data integrity checks for the OpenMIC dataset.

Example
-------
$ ./scripts/verify_dataset.py path/to/openmic path/to/checksums
'''

import argparse
import datetime
import glob
from joblib import Parallel, delayed
import json
import numpy as np
import os
import pandas as pd
import soundfile as sf
import sys
import warnings

import openmic.util


def hash_one(fname):
    fkey = openmic.util.filebase(fname)
    fhash = openmic.util.md5_file(fname)
    return (fkey, fhash)


def hash_collection(filepaths, n_jobs, verbose):
    dfx = delayed(hash_one)
    pool = Parallel(n_jobs=n_jobs, verbose=verbose)

    return pd.DataFrame.from_dict(dict(pool(dfx(fn) for fn in filepaths)),
                                  columns=['md5'], orient='index')


def check_md5(filepaths, checksum_file, n_jobs, verbose):
    df = hash_collection(filepaths, n_jobs, verbose)

    exp = pd.read_csv(checksum_file, index_col=0)
    df = df.join(exp, how='outer', rsuffix='_expected')
    if len(df) != len(filepaths):
        raise warnings.warn('File key mismatch! {} total rows, expected {}'
                            .format(len(df), len(filepaths)))

    success = (df.md5 == df.md5_expected).all()
    if not success:
        percent = 100 * (df.md5 != df.md5_expected).mean()
        raise warnings.warn('MD5 hash mismatch on {:0.2f}% of records'
                            .format(percent))

    return success


def _check_duration(fname, expected_duration, tolerance):
    dur = sf.info(fname).duration
    success = np.abs(dur - expected_duration) < tolerance
    if not success:
        raise warnings.warn('{} does not have the expected duration: {} !~ {}'
                            .format(fname, dur, expected_duration))
    return success


def check_durations(filepaths, expected_duration, n_jobs, verbose, tolerance=0.01):
    dfx = delayed(_check_duration)
    pool = Parallel(n_jobs=n_jobs, verbose=verbose)
    return all(pool(dfx(fn, expected_duration, tolerance) for fn in filepaths))


def _check_shape(json_file, expected_shapes):
    with open(json_file, 'r') as fp:
        data = json.load(fp)

    success = True
    for key, shape in expected_shapes.items():
        act_shape = np.array(data[key]).shape
        if act_shape != shape:
            raise warnings.warn('{}:{} has mismatched shapes: {} != {}'
                                .format(json_file, key, act_shape, shape))
            success &= False

    return success


def check_shapes(filepaths, expected_shapes, n_jobs, verbose):
    dfx = delayed(_check_shape)
    pool = Parallel(n_jobs=n_jobs, verbose=verbose)
    return all(pool(dfx(fn, expected_shapes) for fn in filepaths))


def verify_audio(audio_files, checksum_file, expected_duration,
                 n_jobs=1, verbose=0):
    success = True
    success &= check_md5(audio_files, checksum_file, n_jobs, verbose)
    success &= check_durations(audio_files, expected_duration, n_jobs=n_jobs,
                               verbose=verbose, tolerance=0.01)
    return success


def verify_vggish(vggish_files, checksum_file, expected_shapes,
                  n_jobs=1, verbose=0):

    success = True
    success &= check_md5(vggish_files, checksum_file, n_jobs, verbose)
    success &= check_shapes(vggish_files, expected_shapes,
                            n_jobs=n_jobs, verbose=verbose)
    return success


def verify_labels(label_file, checksum):

    _, act_checksum = hash_one(label_file)
    return act_checksum == checksum


def main(openmic_dir, checksum_dir, n_jobs, verbose):
    success = True
    print("[{}] Verifying Audio...".format(datetime.datetime.now()))
    success &= verify_audio(
        glob.glob(os.path.join(openmic_dir, 'audio/*/*.ogg')),
        os.path.join(checksum_dir, 'openmic-2018-audio.csv'),
        expected_duration=10.0,
        n_jobs=n_jobs, verbose=verbose)

    print("[{}] Verifying VGGish...".format(datetime.datetime.now()))
    success &= verify_vggish(
        glob.glob(os.path.join(openmic_dir, 'vggish/*/*.json')),
        os.path.join(checksum_dir, 'openmic-2018-vggish.csv'),
        expected_shapes=dict(time_points=(10,), features=(10, 128)),
        n_jobs=n_jobs, verbose=verbose)

    print("[{}] Verifying labels...".format(datetime.datetime.now()))
    success &= verify_labels(
        os.path.join(openmic_dir, 'openmic-20k-sparse-labels.csv'),
        '3bbc4f1941fb526d1c9c86b9ece667e7')

    return success


def process_args(args):

    parser = argparse.ArgumentParser(
        description='Verify that the openmic dataset meets expectations.')

    parser.add_argument(dest='openmic_dir', type=str, action='store',
                        help='Path to the uncompressed openmic dataset.')
    parser.add_argument(dest='checksum_dir', type=str, action='store',
                        help='Path to a directory of checksum CSV files.')
    parser.add_argument('--n_jobs', default=-1, type=int,
                        help='Number of cores to use in parallel.')
    parser.add_argument('--verbose', default=0, type=int,
                        help='Verbosity level for processing.')
    return parser.parse_args(args)


if __name__ == '__main__':
    args = process_args(sys.argv[1:])

    success = main(args.openmic_dir, args.checksum_dir,
                   args.n_jobs, args.verbose)
    sys.exit(0 if success else 1)
