from __future__ import annotations

import numpy as np

from music_eval.audio import read_wav


def test_reads_stereo_pcm16(tmp_path, write_wav):
    time = np.arange(8000) / 8000
    samples = np.column_stack(
        (0.25 * np.sin(2 * np.pi * 220 * time), 0.25 * np.sin(2 * np.pi * 330 * time))
    )
    path = write_wav(tmp_path / "stereo.wav", samples)

    audio = read_wav(path)

    assert audio.sample_rate == 8000
    assert audio.channels == 2
    assert audio.frames == 8000
    assert audio.duration_seconds == 1.0
    assert np.max(np.abs(audio.samples - samples)) < 1 / 32768 + 1e-6
    assert audio.samples.flags.writeable is False


def test_rejects_non_wav(tmp_path):
    path = tmp_path / "broken.wav"
    path.write_text("not audio")

    try:
        read_wav(path)
    except ValueError as error:
        assert "cannot decode WAV" in str(error)
    else:
        raise AssertionError("broken WAV should fail")
