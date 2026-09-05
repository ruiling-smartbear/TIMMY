from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class AudioData:
    samples: NDArray[np.float64]
    sample_rate: int
    channels: int
    sample_width_bytes: int

    @property
    def frames(self) -> int:
        return int(self.samples.shape[0])

    @property
    def duration_seconds(self) -> float:
        return self.frames / self.sample_rate


def _decode_pcm(raw: bytes, sample_width: int) -> NDArray[np.float64]:
    if sample_width == 1:
        values_8 = np.frombuffer(raw, dtype=np.uint8).astype(np.float64)
        return (values_8 - 128.0) / 128.0
    if sample_width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    if sample_width == 3:
        packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        values_24 = (
            packed[:, 0].astype(np.int32)
            | (packed[:, 1].astype(np.int32) << 8)
            | (packed[:, 2].astype(np.int32) << 16)
        )
        values_24 = np.where(
            values_24 & 0x800000, values_24 - 0x1000000, values_24
        )
        return values_24.astype(np.float64) / 8388608.0
    if sample_width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    raise ValueError(f"unsupported PCM sample width: {sample_width} bytes")


def read_wav(path: Path) -> AudioData:
    try:
        with wave.open(str(path), "rb") as wav:
            channels = wav.getnchannels()
            sample_rate = wav.getframerate()
            sample_width = wav.getsampwidth()
            frame_count = wav.getnframes()
            compression = wav.getcomptype()
            raw = wav.readframes(frame_count)
    except (OSError, EOFError, wave.Error) as error:
        raise ValueError(f"cannot decode WAV: {error}") from error

    if compression != "NONE":
        raise ValueError(f"unsupported WAV compression: {compression}")
    if channels <= 0 or sample_rate <= 0:
        raise ValueError("WAV has invalid channel count or sample rate")

    decoded = _decode_pcm(raw, sample_width)
    if decoded.size % channels:
        raise ValueError("PCM sample count is not divisible by channel count")
    samples = decoded.reshape(-1, channels)
    samples.setflags(write=False)
    return AudioData(
        samples=samples,
        sample_rate=sample_rate,
        channels=channels,
        sample_width_bytes=sample_width,
    )
