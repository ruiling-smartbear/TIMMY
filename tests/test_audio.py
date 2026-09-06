from __future__ import annotations

import struct

import numpy as np
import pytest

from music_eval.audio import read_wav

PCM_GUID = bytes.fromhex("0100000000001000800000aa00389b71")
FLOAT_GUID = bytes.fromhex("0300000000001000800000aa00389b71")
WAVE_FORMAT_PCM = 1
WAVE_FORMAT_IEEE_FLOAT = 3
WAVE_FORMAT_MULAW = 7
WAVE_FORMAT_EXTENSIBLE = 0xFFFE


def _chunk(chunk_id: bytes, payload: bytes) -> bytes:
    pad = b"\x00" if len(payload) % 2 else b""
    return chunk_id + struct.pack("<I", len(payload)) + payload + pad


def _fmt(
    format_tag: int, channels: int, sample_rate: int, bits: int, subformat: bytes | None = None
) -> bytes:
    block_align = channels * ((bits + 7) // 8)
    payload = struct.pack(
        "<HHIIHH", format_tag, channels, sample_rate, sample_rate * block_align, block_align, bits
    )
    if subformat is not None:
        payload += struct.pack("<HHI", 22, bits, 0) + subformat
    return _chunk(b"fmt ", payload)


def _wav(
    payload: bytes,
    *,
    format_tag: int = WAVE_FORMAT_PCM,
    channels: int = 1,
    sample_rate: int = 8000,
    bits: int = 16,
    subformat: bytes | None = None,
    before_data: bytes = b"",
    after_data: bytes = b"",
    magic: bytes = b"RIFF",
) -> bytes:
    body = b"WAVE" + _fmt(format_tag, channels, sample_rate, bits, subformat)
    body += before_data + _chunk(b"data", payload) + after_data
    return magic + struct.pack("<I", len(body)) + body


def _pack_24bit(codes: np.ndarray) -> bytes:
    return b"".join(int(code).to_bytes(3, "little", signed=True) for code in codes)


def _read(tmp_path, content: bytes):
    path = tmp_path / "hand-built.wav"
    path.write_bytes(content)
    return read_wav(path)


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
    assert audio.sample_format == "pcm"


def test_rejects_non_wav(tmp_path):
    path = tmp_path / "broken.wav"
    path.write_text("not audio")

    try:
        read_wav(path)
    except ValueError as error:
        assert "cannot decode WAV" in str(error)
    else:
        raise AssertionError("broken WAV should fail")


def test_reads_plain_pcm16_exactly(tmp_path):
    codes = np.array([[0, -32768], [32767, 1], [-1, 12345]], dtype="<i2")

    audio = _read(tmp_path, _wav(codes.tobytes(), channels=2, sample_rate=44100))

    assert (audio.sample_format, audio.sample_width_bytes) == ("pcm", 2)
    assert (audio.channels, audio.sample_rate, audio.frames) == (2, 44100, 3)
    assert np.array_equal(audio.samples, codes / 32768.0)


def test_reads_extensible_pcm24_via_subformat_guid(tmp_path):
    codes = np.array([0, 1, -1, 8388607, -8388608, 0x123456, -0x123456])
    content = _wav(
        _pack_24bit(codes),
        format_tag=WAVE_FORMAT_EXTENSIBLE,
        bits=24,
        sample_rate=96000,
        subformat=PCM_GUID,
    )

    audio = _read(tmp_path, content)

    assert (audio.sample_format, audio.sample_width_bytes) == ("pcm", 3)
    assert audio.sample_rate == 96000
    assert np.array_equal(audio.samples[:, 0], codes / 8388608.0)


def test_reads_pcm8_and_pcm32_exactly(tmp_path):
    codes_8 = np.array([0, 128, 255, 64], dtype=np.uint8)
    codes_32 = np.array([0, 2147483647, -2147483648, 65536], dtype="<i4")

    audio_8 = _read(tmp_path, _wav(codes_8.tobytes(), bits=8))
    audio_32 = _read(tmp_path, _wav(codes_32.tobytes(), bits=32))

    assert (audio_8.sample_width_bytes, audio_32.sample_width_bytes) == (1, 4)
    assert np.array_equal(audio_8.samples[:, 0], (codes_8.astype(np.float64) - 128.0) / 128.0)
    assert np.array_equal(audio_32.samples[:, 0], codes_32 / 2147483648.0)


def test_reads_plain_float32_without_clipping(tmp_path):
    values = np.array([0.0, 0.5, -0.25, 1.5, -2.0, 1e-7], dtype="<f4")

    audio = _read(tmp_path, _wav(values.tobytes(), format_tag=WAVE_FORMAT_IEEE_FLOAT, bits=32))

    assert (audio.sample_format, audio.sample_width_bytes) == ("float", 4)
    assert audio.samples.dtype == np.float64
    assert np.max(np.abs(audio.samples[:, 0] - values)) <= 1e-7
    assert audio.samples[:, 0].max() == 1.5


def test_reads_extensible_float32_via_subformat_guid(tmp_path):
    values = np.array([[0.125, -0.125], [3.0, -3.0]], dtype="<f4")
    content = _wav(
        values.tobytes(),
        format_tag=WAVE_FORMAT_EXTENSIBLE,
        channels=2,
        bits=32,
        subformat=FLOAT_GUID,
    )

    audio = _read(tmp_path, content)

    assert (audio.sample_format, audio.channels) == ("float", 2)
    assert np.max(np.abs(audio.samples - values)) <= 1e-7


def test_reads_float64(tmp_path):
    values = np.array([0.1, -0.2, 1.75, -1e-9], dtype="<f8")

    audio = _read(tmp_path, _wav(values.tobytes(), format_tag=WAVE_FORMAT_IEEE_FLOAT, bits=64))

    assert (audio.sample_format, audio.sample_width_bytes) == ("float", 8)
    assert np.max(np.abs(audio.samples[:, 0] - values)) <= 1e-7


def test_skips_list_and_fact_chunks_before_data(tmp_path):
    codes = np.array([100, -100, 200], dtype="<i2")
    info = _chunk(b"LIST", b"INFO" + _chunk(b"ISFT", b"hand-built"))
    fact = _chunk(b"fact", struct.pack("<I", 3))

    audio = _read(tmp_path, _wav(codes.tobytes(), before_data=info + fact))

    assert np.array_equal(audio.samples[:, 0], codes / 32768.0)


def test_honors_pad_byte_of_odd_sized_chunks(tmp_path):
    codes = np.array([0, 128, 255], dtype=np.uint8)
    odd_chunk = _chunk(b"PEAK", b"xyz")
    assert len(odd_chunk) == 12, "a 3-byte payload must be padded to 4"
    content = _wav(
        codes.tobytes(), bits=8, before_data=odd_chunk, after_data=_chunk(b"LIST", b"INFO")
    )

    audio = _read(tmp_path, content)

    assert audio.frames == 3
    assert np.array_equal(audio.samples[:, 0], (codes.astype(np.float64) - 128.0) / 128.0)


def test_truncated_data_chunk_is_a_decode_error(tmp_path):
    complete = _wav(np.zeros(100, dtype="<i2").tobytes())

    with pytest.raises(ValueError, match=r"cannot decode WAV: data chunk is truncated"):
        _read(tmp_path, complete[:-37])


def test_rf64_header_is_rejected(tmp_path):
    content = _wav(np.zeros(4, dtype="<i2").tobytes(), magic=b"RF64")

    with pytest.raises(ValueError, match=r"cannot decode WAV: RF64"):
        _read(tmp_path, content)


def test_mulaw_format_tag_is_rejected(tmp_path):
    content = _wav(bytes(range(8)), format_tag=WAVE_FORMAT_MULAW, bits=8)

    with pytest.raises(ValueError, match=r"cannot decode WAV: unsupported format tag 0x0007"):
        _read(tmp_path, content)


def test_unknown_extensible_subformat_is_rejected(tmp_path):
    content = _wav(bytes(8), format_tag=WAVE_FORMAT_EXTENSIBLE, bits=16, subformat=bytes(16))

    with pytest.raises(ValueError, match=r"cannot decode WAV: unsupported extensible sub-format"):
        _read(tmp_path, content)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"RIFF\x04\x00\x00\x00WAVE", "missing fmt chunk"),
        (b"RIFF\x04\x00\x00\x00WAVE" + _chunk(b"fmt ", bytes(16)), "missing data chunk"),
        (
            b"RIFF\x04\x00\x00\x00WAVE" + _chunk(b"fmt ", bytes(8)) + _chunk(b"data", b""),
            "fmt chunk is shorter",
        ),
        (_wav(bytes(4), channels=0), "invalid channel count"),
        (_wav(bytes(4), format_tag=WAVE_FORMAT_IEEE_FLOAT, bits=16), "float sample width"),
        (_wav(bytes(6), channels=2), "whole number of frames"),
        (_wav(bytes(4), format_tag=WAVE_FORMAT_EXTENSIBLE, bits=16), "extensible fmt chunk"),
    ],
)
def test_malformed_headers_are_decode_errors(tmp_path, content, message):
    with pytest.raises(ValueError, match=f"cannot decode WAV: .*{message}"):
        _read(tmp_path, content)


def test_integrity_reports_sample_format(tmp_path):
    from music_eval.evaluator import evaluate_entry
    from music_eval.models import ManifestEntry

    values = np.full(8000, 0.1, dtype="<f4")
    path = tmp_path / "float.wav"
    path.write_bytes(_wav(values.tobytes(), format_tag=WAVE_FORMAT_IEEE_FLOAT, bits=32))

    integrity = evaluate_entry(ManifestEntry(id="float", audio=path)).metrics["integrity"]

    assert integrity["sample_format"] == "float"
    assert integrity["sample_width_bytes"] == 4
