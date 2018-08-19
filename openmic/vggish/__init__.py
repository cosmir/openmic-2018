import pkg_resources

import os

from openmic.util import md5_file
from .params import *


MODEL_PARAMS = pkg_resources.resource_filename(
    __name__, '__model__/vggish_model.ckpt')
PCA_PARAMS = pkg_resources.resource_filename(
    __name__, '__model__/vggish_pca_params.npz')

for fname in MODEL_PARAMS, PCA_PARAMS:
    if not os.path.exists(fname):
        raise RuntimeError('### VGGish model not found ###\n'
                           '\t >>> {}\n'
                           'Did you forget to run `./scripts/download-deps.sh`?\n'
                           .format(fname))

    fbase = os.path.basename(fname)
    if md5_file(fname) != MD5_CHECKSUMS[fbase]:
        raise RuntimeError(
            '### VGGish model checksums do not match! ###\n\n'
            'Re-run `./scripts/download-deps.sh`, and open an issue at \n'
            'https://github.com/cosmir/openmic-2018/issues/new if that \n'
            'doesn\'t resolve the problem.\n')
