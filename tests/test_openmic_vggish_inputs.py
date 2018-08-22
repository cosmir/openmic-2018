import pytest

import openmic.vggish.inputs as inputs


def test_soundfile_to_examples(ogg_file):
    examples = inputs.soundfile_to_examples(ogg_file)
    assert examples is not None
    assert len([_ for _ in examples]) > 0


def test_soundfile_to_examples_bad_format(mp3_file):
    with pytest.raises(RuntimeError):
        inputs.soundfile_to_examples(mp3_file)


def test_soundfile_to_examples_empty_file(empty_audio_file):
    with pytest.raises(ValueError):
        inputs.soundfile_to_examples(empty_audio_file)
