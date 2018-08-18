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
import soundfile as sf
import tensorflow as tf
from tqdm import tqdm
import warnings

import openmic.audioset as audioset
import openmic.audioset.util as util
import openmic.audioset.vggish_input as vggish_input
import openmic.audioset.vggish_slim as vggish_slim
import openmic.audioset.vggish_postprocess as vggish_postprocess


def load_input(filename, strict=False):
    """Load an input audio file as TF-ready examples.

    Parameters
    ----------
    filename : str
        Path to an audio file on disk. Librosa / audioread will try their best
        to read whatever format you throw at it.

    strict : bool, default=False
        If True, raise any errors caught on load; otherwise, will return None
        for upstream handling.

    Returns
    -------
    examples : iterable of tf.Examples, or None on failure
        Audio examples, or None on failure with strict=False.

    Raises
    ------
    audioop.error - Unable to parse audio bitdepth (24bit wavs cough)
    ValueError - Audio file is empty.
    """
    examples = None
    try:
        y, sr = sf.read(filename, always_2d=True)
        # Mono only, `waveform_to_examples` will take care of samplerate
        y = y.mean(axis=-1)

    except BaseException as derp:
        warnings.warn("Unable to parse({}), skipping: {}"
                      .format(filename, derp))
    else:
        if len(y) > 0:
            y = util.normalize(y)
            examples = vggish_input.waveform_to_examples(y, sr)
        elif strict:
            raise ValueError("Audio file is empty: {}".format(filename))
        else:
            warnings.warn("Caught an empty audio file ({}), skipping."
                          .format(filename))
    finally:
        return examples


def main(files_in, outpath):

    pproc = vggish_postprocess.Postprocessor(audioset.PCA_PARAMS)
    success = []
    with tf.Graph().as_default(), tf.Session() as sess:

        vggish_slim.define_vggish_slim(training=False)
        vggish_slim.load_vggish_slim_checkpoint(
            sess, audioset.MODEL_PARAMS)
        features_tensor = sess.graph.get_tensor_by_name(
            audioset.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(
            audioset.OUTPUT_TENSOR_NAME)

        for file_in in tqdm(files_in):

            file_out = os.path.join(
                outpath,
                os.path.extsep.join([os.path.basename(file_in), 'npz']))
            input_data = load_input(file_in)

            if input_data is not None:
                [embedding] = sess.run([embedding_tensor],
                                       feed_dict={features_tensor: input_data})

                emb_pca = pproc.postprocess(embedding)

                np.savez(file_out, time=np.arange(len(embedding)),
                         features=embedding, features_z=emb_pca)

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
