#!/usr/bin/env python
# coding: utf8
'''Compute a Numpy friendly data-structures from VGGish files
Example
-------
$ cd {repo_root}
$ python ./scripts/helper_numpy.py --csv_file /path/to/sparse-labels.csv \
--vggish_path /path/to/vggish/ --output_file /path/to/output.npz

Any valid output file '*.npz' has to be speficied by the user
'''

from __future__ import print_function
import argparse
import numpy as np
import os
import pandas as pd
import sys
import json


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

    inst_lab = np.copy(df)
    for inst in inst_lab:
        x_pos = int(np.arange(songs_num)[sample_key == inst[0]])
        y_pos = int(np.arange(inst_num)[instruments == inst[1]])
        Y_true[x_pos, y_pos] = inst[2]
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
