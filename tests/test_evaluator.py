from __future__ import annotations

from pathlib import Path

import numpy as np

from music_eval.evaluator import evaluate_entry
from music_eval.models import Expectations, ManifestEntry


def _tone(seconds: float, sample_rate: int = 8000) -> np.ndarray:
    time = np.arange(round(seconds * sample_rate)) / sample_rate
    return 0.25 * np.sin(2 * np.pi * 220 * time)


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
    assert result.dropouts[0].end_seconds == 2.0


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
