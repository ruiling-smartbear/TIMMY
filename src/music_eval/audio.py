from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

WAVE_FORMAT_PCM = 0x0001
WAVE_FORMAT_IEEE_FLOAT = 0x0003
WAVE_FORMAT_EXTENSIBLE = 0xFFFE
_FORMAT_NAMES = {WAVE_FORMAT_PCM: "pcm", WAVE_FORMAT_IEEE_FLOAT: "float"}
# KSDATAFORMAT_SUBTYPE_{PCM,IEEE_FLOAT} GUIDs as they are laid out on disk.
_SUBFORMAT_TAGS = {
    bytes.fromhex("0100000000001000800000aa00389b71"): WAVE_FORMAT_PCM,
    bytes.fromhex("0300000000001000800000aa00389b71"): WAVE_FORMAT_IEEE_FLOAT,
}
_SAMPLE_WIDTHS = {"pcm": (1, 2, 3, 4), "float": (4, 8)}


@dataclass(frozen=True)
class AudioData:
    samples: NDArray[np.float64]
    sample_rate: int
    channels: int
    sample_width_bytes: int
    sample_format: str = "pcm"

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


def _decode_float(raw: bytes, sample_width: int) -> NDArray[np.float64]:
    if sample_width not in _SAMPLE_WIDTHS["float"]:
        raise ValueError(f"unsupported float sample width: {sample_width} bytes")
    return np.frombuffer(raw, dtype=f"<f{sample_width}").astype(np.float64)


def _wav_chunks(data: bytes) -> tuple[bytes, bytes]:
    """Return the ``fmt `` and ``data`` chunk payloads of a RIFF/WAVE file."""
    if data[:4] == b"RF64":
        raise ValueError("RF64 files are not supported")
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("missing RIFF/WAVE header")
    chunks: dict[bytes, bytes] = {}
    offset = 12
    while offset + 8 <= len(data):
        chunk_id, size = struct.unpack_from("<4sI", data, offset)
        offset += 8
        if chunk_id in (b"fmt ", b"data") and chunk_id not in chunks:
            if offset + size > len(data):
                raise ValueError(
                    f"{chunk_id.decode('ascii').strip()} chunk is truncated: declares {size} "
                    f"bytes but only {len(data) - offset} remain"
                )
            chunks[chunk_id] = data[offset : offset + size]
        # Unknown chunks (LIST, fact, PEAK, ...) are skipped; odd sizes carry a pad byte.
        offset += size + (size & 1)
    for chunk_id in (b"fmt ", b"data"):
        if chunk_id not in chunks:
            raise ValueError(f"missing {chunk_id.decode('ascii').strip()} chunk")
    return chunks[b"fmt "], chunks[b"data"]


def _wav_format(fmt: bytes) -> tuple[str, int, int, int]:
    """Return (sample_format, channels, sample_rate, sample_width_bytes)."""
    if len(fmt) < 16:
        raise ValueError("fmt chunk is shorter than 16 bytes")
    format_tag, channels, sample_rate, _, _, bits_per_sample = struct.unpack_from("<HHIIHH", fmt)
    if format_tag == WAVE_FORMAT_EXTENSIBLE:
        if len(fmt) < 40:
            raise ValueError("extensible fmt chunk is shorter than 40 bytes")
        subformat = fmt[24:40]
        if subformat not in _SUBFORMAT_TAGS:
            raise ValueError(f"unsupported extensible sub-format {subformat.hex()}")
        format_tag = _SUBFORMAT_TAGS[subformat]
    if format_tag not in _FORMAT_NAMES:
        raise ValueError(
            f"unsupported format tag {format_tag:#06x} (only PCM and IEEE float are supported)"
        )
    sample_format = _FORMAT_NAMES[format_tag]
    if channels <= 0 or sample_rate <= 0:
        raise ValueError("invalid channel count or sample rate")
    sample_width = (bits_per_sample + 7) // 8
    if sample_width not in _SAMPLE_WIDTHS[sample_format]:
        raise ValueError(f"unsupported {sample_format} sample width: {sample_width} bytes")
    return sample_format, channels, sample_rate, sample_width


def read_wav(path: Path) -> AudioData:
    try:
        fmt_chunk, data_chunk = _wav_chunks(path.read_bytes())
        sample_format, channels, sample_rate, sample_width = _wav_format(fmt_chunk)
        if len(data_chunk) % (channels * sample_width):
            raise ValueError("data chunk size is not a whole number of frames")
        decode = _decode_pcm if sample_format == "pcm" else _decode_float
        decoded = decode(data_chunk, sample_width)
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot decode WAV: {error}") from error

    samples = decoded.reshape(-1, channels)
    samples.setflags(write=False)
    return AudioData(
        samples=samples,
        sample_rate=sample_rate,
        channels=channels,
        sample_width_bytes=sample_width,
        sample_format=sample_format,
    )
