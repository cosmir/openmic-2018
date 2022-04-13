import pytest

import os

import openmic.util as util


def test_filebase():
    assert util.filebase('foo/bar.baz') == 'bar'
    assert util.filebase('foo/bar.baz.whiz') == 'bar.baz'
    assert util.filebase('foo/') == ''
    assert util.filebase('') == ''


def test_safe_makedirs(tmpdir):
    util.safe_makedirs(os.path.join(str(tmpdir), 'foo'))
    util.safe_makedirs(os.path.join(str(tmpdir), 'foo'))
    util.safe_makedirs('')
