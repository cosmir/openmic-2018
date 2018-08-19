
from .params import *
from .inputs import waveform_to_examples, soundfile_to_examples
from .postprocessor import Postprocessor
from .model import transform

__pproc__ = Postprocessor(PCA_PARAMS)
postprocess = __pproc__.postprocess
