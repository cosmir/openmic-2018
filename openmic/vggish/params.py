# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Global parameters for the VGGish model.

See vggish_slim.py for more information.
"""
import os
import pkg_resources
from ..util import md5_file

# Architectural constants.
NUM_FRAMES = 96  # Frames in input mel-spectrogram patch.
NUM_BANDS = 64  # Frequency bands in input mel-spectrogram patch.
EMBEDDING_SIZE = 128  # Size of embedding layer.

# Hyperparameters used in feature and example generation.
SAMPLE_RATE = 16000
STFT_WINDOW_LENGTH_SECONDS = 0.025
STFT_HOP_LENGTH_SECONDS = 0.010
NUM_MEL_BINS = NUM_BANDS
MEL_MIN_HZ = 125
MEL_MAX_HZ = 7500
LOG_OFFSET = 0.01  # Offset used for stabilized log of input mel-spectrogram.
EXAMPLE_WINDOW_SECONDS = 0.96  # Each example contains 96 10ms frames
EXAMPLE_HOP_SECONDS = 0.96     # with zero overlap.

# Parameters used for embedding postprocessing.
PCA_EIGEN_VECTORS_NAME = 'pca_eigen_vectors'
PCA_MEANS_NAME = 'pca_means'
QUANTIZE_MIN_VAL = -2.0
QUANTIZE_MAX_VAL = +2.0

# Hyperparameters used in training.
INIT_STDDEV = 0.01  # Standard deviation used to initialize weights.
LEARNING_RATE = 1e-4  # Learning rate for the Adam optimizer.
ADAM_EPSILON = 1e-8  # Epsilon for the Adam optimizer.

# Names of ops, tensors, and features.
INPUT_OP_NAME = 'vggish/input_features'
INPUT_TENSOR_NAME = INPUT_OP_NAME + ':0'
OUTPUT_OP_NAME = 'vggish/embedding'
OUTPUT_TENSOR_NAME = OUTPUT_OP_NAME + ':0'
AUDIO_EMBEDDING_FEATURE_NAME = 'audio_embedding'

START_TIME = 'start_time_seconds'
VIDEO_ID = 'video_id'
LABELS = 'labels'
TIME = 'time'

MD5_CHECKSUMS = {
    'vggish_model.ckpt': 'd1c7011e6366aa34176bb05c705e31a8',
    'vggish_pca_params.npz': 'c80cae691033abe7c7ecd11ea39fc834'
}

MODEL_PARAMS = pkg_resources.resource_filename(
    __name__, '_model/vggish_model.ckpt')
PCA_PARAMS = pkg_resources.resource_filename(
    __name__, '_model/vggish_pca_params.npz')

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
