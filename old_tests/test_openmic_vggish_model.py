import pytest

import numpy as np
import soundfile as sf
import tensorflow as tf

import openmic.vggish.inputs
import openmic.vggish.model as model
from openmic.vggish import waveform_to_features


def test_model_transform_soundfile(ogg_file):
    examples = openmic.vggish.inputs.soundfile_to_examples(ogg_file)
    with tf.Graph().as_default(), tf.Session() as sess:
        time_points, features = model.transform(examples, sess)

    assert len(time_points) == len(features) > 1


def test_wf_to_features(ogg_file):
    data, rate = sf.read(ogg_file)

    time_points_z, features_z = waveform_to_features(data, rate, compress=True)
    assert len(time_points_z) == len(features_z)

    time_points, features = waveform_to_features(data, rate, compress=False)
    assert len(time_points) == len(features)

    assert np.allclose(time_points, time_points_z)

    assert np.allclose(features_z, openmic.vggish.postprocess(features))
