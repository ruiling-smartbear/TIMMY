from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from music_eval.audio import AudioData
from music_eval.models import Finding, ManifestEntry
from music_eval.plugins.base import AnalysisConfig, MetricOutput


class PairwiseFidelityMetric:
    name = "pairwise"

    def evaluate(
        self,
        entry: ManifestEntry,
        candidate: AudioData,
        reference: AudioData | None,
        config: AnalysisConfig,
    ) -> MetricOutput:
        del config
        if reference is None:
            return MetricOutput()

        duration_delta = abs(candidate.duration_seconds - reference.duration_seconds)
        candidate_rms = _rms(candidate.samples)
        reference_rms = _rms(reference.samples)
        exact = bool(
            candidate.sample_rate == reference.sample_rate
            and candidate.samples.shape == reference.samples.shape
            and np.array_equal(candidate.samples, reference.samples)
        )
        metrics: dict[str, Any] = {
            "reference_duration_seconds": reference.duration_seconds,
            "reference_duration_delta_seconds": duration_delta,
            "reference_sample_rate": reference.sample_rate,
            "reference_channels": reference.channels,
            "reference_pcm_exact": exact,
            "reference_rms_delta_db": _rms_delta_db(candidate_rms, reference_rms),
            "reference_waveform_correlation": None,
            "reference_nrmse": None,
        }
        compatible = (
            candidate.sample_rate == reference.sample_rate
            and candidate.samples.shape == reference.samples.shape
            and candidate.samples.size > 1
        )
        if compatible:
            candidate_flat = candidate.samples.reshape(-1)
            reference_flat = reference.samples.reshape(-1)
            if exact:
                metrics["reference_waveform_correlation"] = 1.0
            elif np.std(candidate_flat) > 0 and np.std(reference_flat) > 0:
                metrics["reference_waveform_correlation"] = float(
                    np.corrcoef(candidate_flat, reference_flat)[0, 1]
                )
            denominator = max(reference_rms, 1e-12)
            metrics["reference_nrmse"] = float(
                np.sqrt(np.mean(np.square(candidate_flat - reference_flat))) / denominator
            )

        findings: list[Finding] = []
        expected = entry.expectations
        if (
            expected.max_reference_duration_delta_seconds is not None
            and duration_delta > expected.max_reference_duration_delta_seconds
        ):
            findings.append(
                Finding(
                    "reference_duration_delta_limit",
                    "failure",
                    f"reference_duration_delta_seconds={duration_delta:.6g} exceeds limit "
                    f"{expected.max_reference_duration_delta_seconds:.6g}",
                )
            )
        correlation = metrics["reference_waveform_correlation"]
        nrmse = metrics["reference_nrmse"]
        requires_aligned_waveform = (
            expected.min_reference_waveform_correlation is not None
            or expected.max_reference_nrmse is not None
        )
        if requires_aligned_waveform and (correlation is None or nrmse is None):
            findings.append(
                Finding(
                    "reference_waveform_incompatible",
                    "failure",
                    "waveform gates require matching sample rate, shape, and non-constant audio",
                )
            )
            return MetricOutput(metrics=metrics, findings=findings)
        if (
            expected.min_reference_waveform_correlation is not None
            and float(correlation) < expected.min_reference_waveform_correlation
        ):
            findings.append(
                Finding(
                    "reference_correlation_limit",
                    "failure",
                    f"reference_waveform_correlation={float(correlation):.6g} is below "
                    f"limit {expected.min_reference_waveform_correlation:.6g}",
                )
            )
        if (
            expected.max_reference_nrmse is not None
            and float(nrmse) > expected.max_reference_nrmse
        ):
            findings.append(
                Finding(
                    "reference_nrmse_limit",
                    "failure",
                    f"reference_nrmse={float(nrmse):.6g} exceeds limit "
                    f"{expected.max_reference_nrmse:.6g}",
                )
            )
        return MetricOutput(metrics=metrics, findings=findings)


def _rms(samples: NDArray[np.float64]) -> float:
    # A zero-frame side has no mean; report 0.0 so the level delta below is None, not NaN.
    return float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0


def _rms_delta_db(candidate_rms: float, reference_rms: float) -> float | None:
    if candidate_rms <= 0 or reference_rms <= 0:
        return None
    return float(20.0 * np.log10(candidate_rms / reference_rms))
