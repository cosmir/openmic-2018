#!/usr/bin/env python
# coding: utf8
'''Convenience utilities for interfacing with the VGGish implementation.
'''

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import tensorflow as tf

from .params import AUDIO_EMBEDDING_FEATURE_NAME, LABELS
from .params import START_TIME, TIME, VIDEO_ID


def bytestring_to_record(example):
    """Convert a serialized tf.SequenceExample to Python-friendly objects.

    Parameters
    ----------
    example : str
        A single serialized tf.SequenceExample

    Returns
    -------
    features : np.array, shape=(n, 128)
        Array of feature coefficients over time (axis=0).

    meta : pd.DataFrame, len=n
        Corresponding labels and metadata for these features.
    """
    rec = tf.train.SequenceExample.FromString(example)
    start_time = rec.context.feature[START_TIME].float_list.value[0]
    vid_id = rec.context.feature[VIDEO_ID].bytes_list.value[0].decode('utf-8')
    labels = list(rec.context.feature[LABELS].int64_list.value)
    data = rec.feature_lists.feature_list[AUDIO_EMBEDDING_FEATURE_NAME]
    features = [b.bytes_list.value for b in data.feature]
    features = np.asarray([np.frombuffer(_[0], dtype=np.uint8)
                           for _ in features])
    if features.ndim == 1:
        raise ValueError("Caught unexpected feature shape: {}"
                         .format(features.shape))

    rows = [{VIDEO_ID: vid_id, LABELS: labels, TIME: np.uint16(start_time + t)}
            for t in range(len(features))]

    return features, pd.DataFrame.from_records(data=rows)


def load_tfrecord(fname, n_jobs=1, verbose=0):
    """Transform a YouTube-8M style tfrecord file to numpy / pandas objects.

    Parameters
    ----------
    fname : str
        Filepath on disk to read.

    n_jobs : int, default=-2
        Number of cores to use, defaults to all but one.

    verbose : int, default=0
        Verbosity level for loading.

    Returns
    -------
    features : np.array, shape=(n_obs, n_coeffs)
        All observations, concatenated together,

    meta : pd.DataFrame
        Table of metadata aligned to the features, indexed by `filebase.idx`
    """
    dfx = delayed(bytestring_to_record)
    pool = Parallel(n_jobs=n_jobs, verbose=verbose)
    results = pool(dfx(x) for x in tf.python_io.tf_record_iterator(fname))
    features = np.concatenate([xy[0] for xy in results], axis=0)
    meta = pd.concat([xy[1] for xy in results], axis=0, ignore_index=True)
    return features, meta
