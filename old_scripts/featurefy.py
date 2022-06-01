#!/usr/bin/env python
# coding: utf8
'''Compute VGGish features for a batch of files

There are two modes of operation, either by passing in a single audio
filepath, or a newline-separated list of filepaths. See below for examples.

Example
-------
$ cd {repo_root}
$ ./scripts/featurefy.py --file /some/audio/file.wav ./output_dir
OR
$ ls /path/to/audio/*wav > file_list.txt
$ ./scripts/featurefy.py --input_list file_list.txt ./output_dir

Each jams file must contain at least one annotation in the `tag_openmic25`
namespace.
'''

import argparse
import numpy as np
import os
import pandas as pd
import sys
import tensorflow as tf
from tqdm import tqdm

from openmic.util import filebase
import openmic.vggish


def main(files_in, outpath):

    success = []
    with tf.Graph().as_default(), tf.Session() as sess:

        for file_in in tqdm(files_in):
            file_out = os.path.join(
                outpath,
                os.path.extsep.join([filebase(file_in), 'npz']))

            try:
                examples = openmic.vggish.soundfile_to_examples(file_in)

                time_points, features = openmic.vggish.transform(examples, sess)
                features_z = openmic.vggish.postprocess(features)

                np.savez(file_out, time=time_points,
                         features=features, features_z=features_z)
            except ValueError as derp:
                pass

            success.append(os.path.exists(file_out))

    return success


def process_args(args):

    parser = argparse.ArgumentParser(description='VGGish feature extractor')

    parser.add_argument('--input_list', default='', type=str,
                        help='Path to a newline separated list of filepaths.')
    parser.add_argument('--file', default='',
                        type=str, help='Path to an audio file to process.')

    parser.add_argument(dest='output_path', type=str, action='store',
                        help='Path to store output files in NPZ format')
    return parser.parse_args(args)


def load_files_in(input_list):

    files_in = pd.read_table(input_list, header=None)
    return list(files_in[0])


if __name__ == '__main__':
    args = process_args(sys.argv[1:])

    if not args.input_list and not args.file:
        raise ValueError("One of `--file` or `--input_list` must be given.")
    elif args.input_list and args.file:
        raise ValueError(
            "Only one of `--file` or `--input_list` can be given.")
    elif args.file:
        files_in = [args.file]
    else:
        files_in = load_files_in(args.input_list)

    success = all(main(files_in, args.output_path))
    sys.exit(0 if success else 1)
