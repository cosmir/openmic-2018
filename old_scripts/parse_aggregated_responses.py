#!/usr/bin/env python
# coding: utf8
'''Script to parse aggregated annotation responses into a CSV file of labels.

Example
-------
$ ./scripts/parse_aggregated_responses.py \
    "path/to/dir/*.csv" \
    openmic-2018-aggregated-labels.csv

'''
from __future__ import print_function

import argparse
import glob
import os
import pandas as pd
import sys
import tqdm

YN_MAP = {'no': -1, 'yes': 1}
COLUMNS = ['sample_key', 'instrument', 'relevance', 'num_responses']
CONF_COL = 'does_this_recording_contain_{}:confidence'
CONTAIN_COL = 'does_this_recording_contain_{}'


def parse_one(row):
    sign = YN_MAP[row[CONTAIN_COL.format(row.instrument)]]
    conf = row[CONF_COL.format(row.instrument)] / 2.0
    proba = 0.5 + sign * conf
    return dict(sample_key=row.sample_key, instrument=row.instrument,
                relevance=proba, num_responses=row._trusted_judgments)


def main(csv_files, output_filename):
    records = []
    for csv_file in tqdm.tqdm(csv_files):
        records += pd.read_csv(csv_file).apply(parse_one, axis=1).values.tolist()

    df = pd.DataFrame.from_records(records)
    print('Loaded {} records'.format(len(df)))

    df.sort_values(by='sample_key', inplace=True)
    df.to_csv(output_filename, columns=COLUMNS, index=None)
    return os.path.exists(output_filename)


def process_args(args):

    parser = argparse.ArgumentParser(description='Aggregated annotation results parser')

    parser.add_argument('csv_pattern', type=str, action='store',
                        help='Glob-style file pattern for picking up CSV files.')
    parser.add_argument(dest='output_filename', type=str, action='store',
                        help='Output filename for writing the sparse label CSV.')
    return parser.parse_args(args)


if __name__ == '__main__':
    args = process_args(sys.argv[1:])

    csv_files = glob.glob(args.csv_pattern)

    success = main(csv_files, args.output_filename)
    sys.exit(0 if success else 1)
