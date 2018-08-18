import pkg_resources

from .vggish_params import *


MODEL_PARAMS = pkg_resources.resource_filename(
    __name__, '.model/vggish_model.cpkt')
PCA_PARAMS = pkg_resources.resource_filename(
    __name__, '.model/vggish_pca_params.npz')
