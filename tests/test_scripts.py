import pytest

import featurefy


def test_featurefy_main(audio_file, tmpdir):
    success = featurefy.main([audio_file], str(tmpdir))
    assert all(success)


def test_featurefy_main_garbage_audio(empty_audio_file, tmpdir):
    success = featurefy.main([empty_audio_file], str(tmpdir))
    assert not all(success)
