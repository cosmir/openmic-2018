import pytest

import tensorflow as tf

import openmic.vggish.inputs
import openmic.vggish.model as model


def test_model_transform_soundfile(ogg_file):
    examples = openmic.vggish.inputs.soundfile_to_examples(ogg_file)
    with tf.Graph().as_default(), tf.Session() as sess:
        time_points, features = model.transform(examples, sess)

    assert len(time_points) == len(features) > 1
