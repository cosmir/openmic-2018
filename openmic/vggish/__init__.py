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
