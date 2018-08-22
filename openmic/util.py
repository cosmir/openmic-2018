#!/usr/bin/env python
# coding: utf8
'''Standalone convenience utilities.'''

import hashlib
import numpy as np
import os


def filebase(fname):
    return os.path.splitext(os.path.basename(fname))[0]


def safe_makedirs(dpath):
    if not os.path.exists(dpath) and dpath:
        os.makedirs(dpath)


def md5_file(fname):
    hsh = hashlib.md5(open(fname, 'rb').read())
    return hsh.hexdigest()


def tiny(x):
    '''Return the tiniest value for a given data type.

    Ported from librosa 0.6
    '''
    x = np.asarray(x)

    if np.issubdtype(x.dtype, np.floating) or np.issubdtype(x.dtype, np.complexfloating):
        dtype = x.dtype
    else:
        dtype = np.float32

    return np.finfo(dtype).tiny


def normalize(S):
    '''Max-scale an input with some guards against numerical instability.

    Ported from librosa 0.6
    '''
    mag = np.abs(S).astype(np.float)

    length = np.max(mag, axis=0, keepdims=True)
    small_idx = length < tiny(S)
    Snorm = np.empty_like(S)

    length[small_idx] = 1.0
    Snorm[:] = S / length
    return Snorm
