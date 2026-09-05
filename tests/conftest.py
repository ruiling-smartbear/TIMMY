from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def write_wav():
    def _write(path: Path, samples: np.ndarray, sample_rate: int = 8000) -> Path:
        if samples.ndim == 1:
            samples = samples[:, None]
        pcm = np.clip(samples, -1.0, 32767 / 32768)
        pcm = np.round(pcm * 32768).astype("<i2")
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(samples.shape[1])
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm.tobytes())
        return path

    return _write
