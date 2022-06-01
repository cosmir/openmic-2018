#!/usr/bin/env python
# coding: utf8
'''Script to parse individual annotator responses into a well-formed CSV file.

Example
-------
$ ./scripts/parse_individual_responses.py \
    "path/to/dir/*.csv" \
    openmic-2018-individual-responses.csv

'''
from __future__ import print_function

import argparse
import glob
import json
import os
import pandas as pd
import sys
import tqdm
import uuid
import warnings

YN_MAP = {'no': 0, 'yes': 1}
OUTPUT_COLUMNS = ['sample_key', 'worker_id', 'worker_trust',
                  'channel', 'instrument', 'response']
CONTAIN_COL = 'does_this_recording_contain_{}'


def parse_one(row):
    '''
    Parameters
    ----------
    row : pd.Series
        Series record with at least the following fields:
          [_channel, _trust, _worker_id, instrument, sample_key,
           does_this_recording_contain_{instrument}]

    Returns
    -------
    resp : dict
        Object with the following fields:
          [sample_key, worker_id, worker_trust, channel, instrument, response]
    '''

    response = YN_MAP.get(row[CONTAIN_COL.format(row.instrument)])
    if response is None:
        warnings.warn("Null response: {}".format(row.tolist()))

    return dict(sample_key=row.sample_key, worker_id=row._worker_id,
                worker_trust=row._trust, channel=row._channel,
                instrument=row.instrument, response=response)


def encrypt_field(values, hashlen=8, retries=5):
    unique_values = set(values)
    for n in range(retries):
        hashmap = {val: str(uuid.uuid4()).replace('-', '')[:hashlen]
                   for val in list(unique_values)}

        unique_hashes = set(hashmap.values())
        if len(unique_hashes) == len(unique_values):
            # No collisions, we're done here.
            break

    if len(unique_hashes) != len(unique_values):
        raise ValueError('hashlen={} has caused collisions, try increasing.'
                         .format(hashlen))

    print('Encrypted {} unique values'.format(len(hashmap)))
    return list(map(hashmap.get, values)), hashmap


def main(csv_files, output_filename, hashfile=None):
    records = []
    for csv_file in tqdm.tqdm(csv_files):
        records += pd.read_csv(csv_file).apply(parse_one, axis=1).values.tolist()

    df = pd.DataFrame.from_records(records)
    print('Loaded {} records'.format(len(df)))

    df.sort_values(by='sample_key', inplace=True)
    worker_ids, worker_hashmap = encrypt_field(df.worker_id.values.tolist(), 8)
    df['worker_id'] = worker_ids
    channels, channel_hashmap = encrypt_field(df.channel.values.tolist(), 4)
    df['channel'] = channels
    df.to_csv(output_filename, columns=OUTPUT_COLUMNS, index=None)

    if hashfile:
        with open(hashfile, 'w') as fp:
            json.dump(dict(worker_ids=worker_hashmap,
                           channels=channel_hashmap), fp)

    return os.path.exists(output_filename)


def process_args(args):

    parser = argparse.ArgumentParser(description='Raw annotation results parser')

    parser.add_argument('csv_pattern', type=str, action='store',
                        help='Glob-style file pattern for picking up CSV files.')
    parser.add_argument(dest='output_filename', type=str, action='store',
                        help='Output filename for writing the sparse label CSV.')
    parser.add_argument('--hashfile', type=str, default='',
                        help='Optional file for writing the hash-mappings to disk.')
    return parser.parse_args(args)


if __name__ == '__main__':
    args = process_args(sys.argv[1:])

    csv_files = glob.glob(args.csv_pattern)

    success = main(csv_files, args.output_filename, args.hashfile)
    sys.exit(0 if success else 1)
