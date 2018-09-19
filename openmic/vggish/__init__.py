'''
VGGish API

Provides a simple interface to the VGGish model:

Inputs
------
 * waveform_to_examples: tf.Examples from an ndarray
 * soundfile_to_examples: tf.Examples from a sound file

Transforms
----------
 * transform: Times and VGGish features (ndarray) from tf.Examples
 * postprocess: PCA'ed embeddings from VGGish features

'''

from .params import *

from .inputs import waveform_to_examples, soundfile_to_examples
from .model import transform
from .postprocessor import Postprocessor

__pproc__ = Postprocessor(PCA_PARAMS)
postprocess = __pproc__.postprocess


def waveform_to_features(data, sample_rate, compress=True):
    '''Converts an audio waveform to VGGish features, with or without
    PCA compression.

    Parameters
    ----------
    data : np.array of either one dimension (mono) or two dimensions (stereo)

    sample_rate:
        Sample rate of the audio data

    compress : bool
        If True, PCA and quantization are applied to the features.
        If False, the features are taken directly from the model output

    Returns
    -------
    time_points : np.ndarray, len=n
        Time points in seconds of the features

    features : np.ndarray, shape=(n, 128)
        The output features, with or without PCA compression and quantization.
    '''

    import tensorflow as tf

    examples = waveform_to_examples(data, sample_rate)

    with tf.Graph().as_default(), tf.Session() as sess:
        time_points, features = transform(examples, sess)

        if compress:
            features_z = postprocess(features)
            return time_points, features_z

        return time_points, features
