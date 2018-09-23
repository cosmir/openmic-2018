#!/usr/bin/env python
# coding: utf8
'''Compute a Numpy friendly data-structures from VGGish files
Example
-------
$ cd {repo_root}
$ python ./scripts/helper_numpy.py \
    --csv_file /path/to/openmic-2018-aggregated-labels.csv \
    --vggish_path /path/to/vggish/ \
    --output_file /path/to/openmic-2018.npz

Any valid output file '*.npz' has to be speficied by the user
The produced file contains four keys:
- X: ndarray, shape=(songs_num, 10, 128), dtype=int8, values in [0, 255]
It contains the VGGish features for an excerpt
- Y_true: ndarray, shape=(number of inputs, number of classes), dtype=np.float32, values in [0, 1])
It contains the relevance of the annotation. 0 means the instrument is
strongly not present. 1 means that the instrument is strongly present
- Y_mask: ndarray, shape=(number of inputs, number of classes), dtype=bool,
It indicates the presence/absence of an annotation
- sample_key: nparray, shape=(number of inputs,), dtype=object, values are strings
It contains the sample key of the song
'''

from __future__ import print_function
import argparse
import json
import numpy as np
import os
import pandas as pd
import sys


def main(csvfile, vggishpath, outfile):

    success = []
    df = pd.read_csv(csvfile)
    instruments = np.unique(df['instrument'])
    sample_key = np.unique(df['sample_key'])
    songs_num = len(sample_key)
    inst_num = len(instruments)

    print('Extracting the vggish features...')
    X = np.empty([songs_num, 10, 128], dtype=int)
    count = 0
    for sk in sample_key:
        sk_prefix = sk[:3]
        full_name = os.path.join(vggishpath, sk_prefix, sk + '.json')
        with open(full_name, 'r') as f:
            X_tmp = json.load(f)
        X[count] = X_tmp['features']
        count += 1

    print('Extracting the labels information...')
    Y_true = 0.5 * np.ones([songs_num, inst_num], dtype=float)
    Y_mask = np.zeros([songs_num, inst_num], dtype=bool)

    for _, row in df.iterrows():
        x_pos = int(np.arange(songs_num)[sample_key == row.sample_key])
        y_pos = int(np.arange(inst_num)[instruments == row.instrument])
        Y_true[x_pos, y_pos] = row.relevance
        Y_mask[x_pos, y_pos] = 1

    print('Saving the NPZ file...')
    np.savez(outfile, X=X, Y_true=Y_true,
             Y_mask=Y_mask, sample_key=sample_key)
    success.append(os.path.exists(outfile))

    print('Done.')
    return success


def process_args(args):

    parser = argparse.ArgumentParser(description='VGGish to NumPy data generator')

    parser.add_argument('--csv_file', default='', type=str,
                        help='Path to the sparse labels CSV file.')
    parser.add_argument('--vggish_path', default='',
                        type=str, help='Path to where the VGGish features are stored.')

    parser.add_argument('--output_file', default='',
                        type=str, help='Path and name to store the inputs files in a single NPZ file')
    return parser.parse_args(args)


if __name__ == '__main__':
    args = process_args(sys.argv[1:])

    if not args.csv_file or not args.vggish_path or not args.output_file:
        raise ValueError("Both `--vggish_path`, `--csv_file` and `--output_file` must be given.")

    success = all(main(args.csv_file, args.vggish_path, args.output_file))
    sys.exit(0 if success else 1)
