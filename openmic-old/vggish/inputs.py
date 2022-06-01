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

"""Compute input examples for VGGish from audio waveform."""

import numpy as np
import resampy
from scipy.io import wavfile
import soundfile as sf
import warnings

from . import mel_features
from . import params
from ..util import normalize


def waveform_to_examples(data, sample_rate):
    """Converts audio waveform into an array of examples for VGGish.

    Args:
        data: np.array of either one dimension (mono) or two dimensions
        (multi-channel, with the outer dimension representing channels).
        Each sample is generally expected to lie in the range [-1.0, +1.0],
        although this is not required.
        sample_rate: Sample rate of data.

    Returns:
        3-D np.array of shape [num_examples, num_frames, num_bands] which
        represents a sequence of examples, each of which contains a patch of
        log mel spectrogram, covering num_frames frames of audio and num_bands
        mel frequency bands, where the frame length is
        params.STFT_HOP_LENGTH_SECONDS.
    """
    # Convert to mono.
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    # Resample to the rate assumed by VGGish.
    if sample_rate != params.SAMPLE_RATE:
        data = resampy.resample(data, sample_rate, params.SAMPLE_RATE)

    # Compute log mel spectrogram features.
    log_mel = mel_features.log_mel_spectrogram(
        data,
        audio_sample_rate=params.SAMPLE_RATE,
        log_offset=params.LOG_OFFSET,
        window_length_secs=params.STFT_WINDOW_LENGTH_SECONDS,
        hop_length_secs=params.STFT_HOP_LENGTH_SECONDS,
        num_mel_bins=params.NUM_MEL_BINS,
        lower_edge_hertz=params.MEL_MIN_HZ,
        upper_edge_hertz=params.MEL_MAX_HZ)

    # Frame features into examples.
    features_sample_rate = 1.0 / params.STFT_HOP_LENGTH_SECONDS
    example_window_length = int(round(
        params.EXAMPLE_WINDOW_SECONDS * features_sample_rate))
    example_hop_length = int(round(
        params.EXAMPLE_HOP_SECONDS * features_sample_rate))
    log_mel_examples = mel_features.frame(
        log_mel,
        window_length=example_window_length,
        hop_length=example_hop_length)
    return log_mel_examples


def wavfile_to_examples(wav_file):
    """Convenience wrapper around waveform_to_examples() for a common WAV
    format.

    Args:
        wav_file: String path to a file, or a file-like object. The file
        is assumed to contain WAV audio data with signed 16-bit PCM samples.

    Returns:
        See waveform_to_examples.
    """
    sr, wav_data = wavfile.read(wav_file)
    if wav_data.dtype != np.int16:
        raise ValueError('Bad sample type: %r' % wav_data.dtype)
    samples = wav_data / 32768.0  # Convert to [-1.0, +1.0]
    return waveform_to_examples(samples, sr)


def soundfile_to_examples(filename):
    """Load a soundfile as TF examples.

    Parameters
    ----------
    filename : str
        Path to an audio file on disk. Librosa / audioread will try their best
        to read whatever format you throw at it.

    Returns
    -------
    examples : iterable of tf.Examples
        Audio examples
    """
    examples = None
    y, sr = sf.read(filename, always_2d=True)
    # Mono only, `waveform_to_examples` will take care of samplerate
    try:
        y = y.mean(axis=-1)
        examples = waveform_to_examples(normalize(y), sr)

    except ValueError as derp:
        warnings.warn('Caught an empty audio file ({}).'.format(filename))
        raise derp

    return examples
