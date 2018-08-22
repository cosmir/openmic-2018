#!/usr/bin/env python
# coding: utf8
'''VGGish transform definitions.'''

import numpy as np

from . import params

from .slim import load_vggish_slim_checkpoint, define_vggish_slim


def transform(examples, sess):
    '''Compute VGGish features for an iterable of examples.

    Parameters
    ----------
    examples : iterable of tf.Examples
        Examples to process by the model.
        See openmic.vggish.inputs.{soundfile_to_examples, waveform_to_examples}

    sess : tf.Session
        Open tensorflow session.

    Returns
    -------
    time_points : np.ndarray, len=n
        Time points in seconds of the feature vector.

    features : np.ndarray, shape=(n, 128), dtype=np.uint8
        VGGish feature array.
    '''
    define_vggish_slim(training=False)
    load_vggish_slim_checkpoint(sess, params.MODEL_PARAMS)

    features_tensor = sess.graph.get_tensor_by_name(params.INPUT_TENSOR_NAME)
    embedding_tensor = sess.graph.get_tensor_by_name(params.OUTPUT_TENSOR_NAME)

    [features] = sess.run([embedding_tensor],
                          feed_dict={features_tensor: examples})

    time_points = np.arange(len(features)) * params.EXAMPLE_HOP_SECONDS

    return time_points, features
