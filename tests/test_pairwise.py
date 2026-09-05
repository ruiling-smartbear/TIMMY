from __future__ import annotations

import numpy as np

from music_eval.evaluator import evaluate_entry
from music_eval.models import Expectations, ManifestEntry


def _tone(seconds: float = 1.0, sample_rate: int = 8000) -> np.ndarray:
    time = np.arange(round(seconds * sample_rate)) / sample_rate
    return 0.25 * np.sin(2 * np.pi * 220 * time)


def test_exact_reference_passes_explicit_waveform_gates(tmp_path, write_wav):
    samples = _tone()
    candidate = write_wav(tmp_path / "candidate.wav", samples)
    reference = write_wav(tmp_path / "reference.wav", samples)
    entry = ManifestEntry(
        id="exact",
        audio=candidate,
        reference=reference,
        expectations=Expectations(
            min_reference_waveform_correlation=0.999,
            max_reference_nrmse=0.0,
            max_reference_duration_delta_seconds=0.0,
        ),
    )

    result = evaluate_entry(entry)

    assert result.status == "pass"
    assert result.metrics["pairwise"]["reference_pcm_exact"] is True
    assert result.metrics["pairwise"]["reference_waveform_correlation"] == 1.0
    assert result.metrics["pairwise"]["reference_nrmse"] == 0.0


def test_reference_regression_fails_correlation_and_nrmse_gates(tmp_path, write_wav):
    candidate = write_wav(tmp_path / "candidate.wav", -_tone())
    reference = write_wav(tmp_path / "reference.wav", _tone())
    entry = ManifestEntry(
        id="changed",
        audio=candidate,
        reference=reference,
        expectations=Expectations(
            min_reference_waveform_correlation=0.9,
            max_reference_nrmse=0.1,
        ),
    )

    result = evaluate_entry(entry)

    assert result.status == "fail"
    assert {finding.code for finding in result.findings} >= {
        "reference_correlation_limit",
        "reference_nrmse_limit",
    }


def test_incompatible_reference_fails_only_when_waveform_gate_requested(
    tmp_path, write_wav
):
    candidate = write_wav(tmp_path / "candidate.wav", _tone())
    reference = write_wav(
        tmp_path / "reference.wav", _tone(sample_rate=16000), sample_rate=16000
    )
    entry = ManifestEntry(
        id="incompatible",
        audio=candidate,
        reference=reference,
        expectations=Expectations(min_reference_waveform_correlation=0.9),
    )

    result = evaluate_entry(entry)

    assert result.status == "fail"
    pairwise_findings = [
        finding for finding in result.findings if finding.source == "pairwise"
    ]
    assert [finding.code for finding in pairwise_findings] == [
        "reference_waveform_incompatible"
    ]


def test_missing_reference_is_reported_as_core_failure(tmp_path, write_wav):
    candidate = write_wav(tmp_path / "candidate.wav", _tone())

    result = evaluate_entry(
        ManifestEntry(
            id="missing-reference",
            audio=candidate,
            reference=tmp_path / "missing.wav",
        )
    )

    assert result.status == "fail"
    assert result.findings[0].code == "reference_wav_decode"
