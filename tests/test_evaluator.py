from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from music_eval.evaluator import evaluate_entry
from music_eval.metrics import detect_dropouts
from music_eval.models import Expectations, ManifestEntry


def _tone(seconds: float, sample_rate: int = 8000) -> np.ndarray:
    time = np.arange(round(seconds * sample_rate)) / sample_rate
    return 0.25 * np.sin(2 * np.pi * 220 * time)


def _tone_with_dropout(
    offset_seconds: float, dropout_seconds: float, sample_rate: int = 8000
) -> np.ndarray:
    samples = _tone(3.0, sample_rate)
    start = round(offset_seconds * sample_rate)
    samples[start : start + round(dropout_seconds * sample_rate)] = 0.0
    return samples


def test_healthy_tone_passes(tmp_path, write_wav):
    path = write_wav(tmp_path / "healthy.wav", _tone(2.0))
    entry = ManifestEntry(
        id="healthy",
        audio=path,
        expectations=Expectations(
            duration_seconds=2.0,
            sample_rate=8000,
            channels=1,
            max_dropout_seconds=0.0,
        ),
    )

    result = evaluate_entry(entry)

    assert result.status == "pass"
    assert result.metrics["integrity"]["duration_seconds"] == 2.0
    assert result.dropouts == []


def test_detects_middle_dropout(tmp_path, write_wav):
    samples = np.concatenate((_tone(1.0), np.zeros(8000), _tone(1.0)))
    path = write_wav(tmp_path / "dropout.wav", samples)

    result = evaluate_entry(ManifestEntry(id="dropout", audio=path))

    assert result.status == "warning"
    assert [finding.code for finding in result.findings] == ["dropout_detected"]
    assert len(result.dropouts) == 1
    assert result.dropouts[0].start_seconds == 1.0
    # The second tone starts at sin(0) == 0, so the true silent span is one sample longer.
    assert result.dropouts[0].end_seconds == pytest.approx(2.0, abs=1e-3)


def test_off_grid_dropout_is_sample_accurate_and_fails_its_gate(tmp_path, write_wav):
    path = write_wav(tmp_path / "off-grid.wav", _tone_with_dropout(0.10, 0.5))
    entry = ManifestEntry(
        id="off-grid",
        audio=path,
        expectations=Expectations(max_dropout_seconds=0.25),
    )

    result = evaluate_entry(entry)

    assert result.status == "fail"
    assert "dropout_limit" in [finding.code for finding in result.findings]
    assert len(result.dropouts) == 1
    assert result.dropouts[0].start_seconds == pytest.approx(0.10, abs=1e-3)
    assert result.dropouts[0].end_seconds == pytest.approx(0.60, abs=1e-3)
    assert result.dropouts[0].duration_seconds == pytest.approx(0.5, abs=1e-3)
    assert result.metrics["integrity"]["longest_dropout_seconds"] == pytest.approx(0.5, abs=1e-3)


@pytest.mark.parametrize("offset", [0.0, 0.03, 0.1, 0.17, 0.24])
def test_half_second_dropout_is_detected_at_every_offset(offset):
    samples = _tone_with_dropout(offset, 0.5)[:, None]

    dropouts, _ratio, longest = detect_dropouts(
        samples, 8000, silence_dbfs=-50.0, window_seconds=0.25, minimum_seconds=0.5
    )

    assert len(dropouts) == 1
    assert dropouts[0].start_seconds == pytest.approx(offset, abs=1e-3)
    assert dropouts[0].end_seconds == pytest.approx(offset + 0.5, abs=1e-3)
    assert longest == pytest.approx(0.5, abs=1e-3)


def test_quiet_but_not_silent_passage_is_not_a_dropout():
    noise = np.random.default_rng(20260906).normal(size=8000)
    noise *= 0.01 / np.sqrt(np.mean(np.square(noise)))  # -40 dBFS RMS
    samples = np.concatenate((_tone(1.0), noise, _tone(1.0)))[:, None]

    dropouts, ratio, longest = detect_dropouts(
        samples, 8000, silence_dbfs=-50.0, window_seconds=0.25, minimum_seconds=0.5
    )

    assert dropouts == []
    assert ratio == 0.0
    assert longest == 0.0


def test_explicit_dropout_limit_fails(tmp_path, write_wav):
    samples = np.concatenate((_tone(1.0), np.zeros(8000), _tone(1.0)))
    path = write_wav(tmp_path / "dropout.wav", samples)
    entry = ManifestEntry(
        id="gated",
        audio=path,
        expectations=Expectations(max_dropout_seconds=0.5),
    )

    result = evaluate_entry(entry)

    assert result.status == "fail"
    assert "dropout_limit" in [finding.code for finding in result.findings]


def test_contract_mismatches_fail(tmp_path, write_wav):
    path = write_wav(tmp_path / "short.wav", _tone(1.0))
    entry = ManifestEntry(
        id="contract",
        audio=path,
        expectations=Expectations(
            duration_seconds=2.0,
            duration_tolerance_seconds=0.1,
            sample_rate=16000,
            channels=2,
        ),
    )

    result = evaluate_entry(entry)

    assert result.status == "fail"
    assert {finding.code for finding in result.findings} >= {
        "duration_mismatch",
        "sample_rate_mismatch",
        "channel_mismatch",
    }


def test_missing_audio_is_a_result_not_a_crash():
    result = evaluate_entry(ManifestEntry(id="missing", audio=Path("missing.wav")))

    assert result.status == "fail"
    assert result.findings[0].code == "wav_decode"
