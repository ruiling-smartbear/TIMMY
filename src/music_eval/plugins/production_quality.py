from __future__ import annotations

import math
from collections.abc import Callable
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from music_eval.audio import AudioData
from music_eval.models import ManifestEntry
from music_eval.plugins.base import AnalysisConfig, MetricOutput

EPSILON = 1e-12
TRUE_PEAK_OVERSAMPLE = 4
CLICK_ABSOLUTE_DELTA = 0.5
CLICK_MAD_MULTIPLIER = 30.0


class ProductionRuntime(Protocol):
    pyloudnorm_version: str
    scipy_version: str

    def integrated_loudness(
        self, samples: NDArray[np.float64], sample_rate: int
    ) -> float: ...

    def loudness_range(
        self, samples: NDArray[np.float64], sample_rate: int
    ) -> float: ...

    def oversample(
        self, samples: NDArray[np.float64], factor: int
    ) -> NDArray[np.float64]: ...

    def welch(
        self, samples: NDArray[np.float64], sample_rate: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...


RuntimeFactory = Callable[[], ProductionRuntime]


def _version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unknown"


class ScipyLoudnessRuntime:
    def __init__(self) -> None:
        try:
            self._pyloudnorm = import_module("pyloudnorm")
            self._signal = import_module("scipy.signal")
        except ImportError as error:
            raise RuntimeError(
                "production quality metrics are optional; install music-eval[production]"
            ) from error
        self.pyloudnorm_version = _version("pyloudnorm")
        self.scipy_version = _version("scipy")

    def integrated_loudness(
        self, samples: NDArray[np.float64], sample_rate: int
    ) -> float:
        return float(self._pyloudnorm.Meter(sample_rate).integrated_loudness(samples))

    def loudness_range(
        self, samples: NDArray[np.float64], sample_rate: int
    ) -> float:
        return float(self._pyloudnorm.Meter(sample_rate).loudness_range(samples))

    def oversample(
        self, samples: NDArray[np.float64], factor: int
    ) -> NDArray[np.float64]:
        return np.asarray(self._signal.resample_poly(samples, factor, 1, axis=0))

    def welch(
        self, samples: NDArray[np.float64], sample_rate: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        segment_frames = min(8192, samples.size)
        frequencies, power = self._signal.welch(
            samples,
            fs=sample_rate,
            nperseg=segment_frames,
            noverlap=segment_frames // 2,
            scaling="spectrum",
        )
        return (
            np.asarray(frequencies, dtype=np.float64),
            np.asarray(power, dtype=np.float64),
        )


def _create_runtime() -> ProductionRuntime:
    return ScipyLoudnessRuntime()


def _db(value: float) -> float:
    return 20.0 * math.log10(max(value, EPSILON))


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _percentiles(values: NDArray[np.float64]) -> dict[str, float]:
    return {
        "minimum": float(np.min(values)),
        "p10": float(np.percentile(values, 10)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "maximum": float(np.max(values)),
    }


def _block_rms_levels(samples: NDArray[np.float64], sample_rate: int) -> NDArray[np.float64]:
    block_frames = max(1, round(0.4 * sample_rate))
    hop_frames = max(1, round(0.1 * sample_rate))
    if samples.shape[0] <= block_frames:
        starts = [0]
    else:
        starts = list(range(0, samples.shape[0] - block_frames + 1, hop_frames))
        final_start = samples.shape[0] - block_frames
        if starts[-1] != final_start:
            starts.append(final_start)
    levels = []
    for start in starts:
        block = samples[start : start + block_frames]
        rms = float(np.sqrt(np.mean(np.square(block)))) if block.size else 0.0
        levels.append(_db(rms))
    return np.asarray(levels, dtype=np.float64)


def _spectral_evidence(
    runtime: ProductionRuntime, mono: NDArray[np.float64], sample_rate: int
) -> dict[str, Any]:
    frequencies, power = runtime.welch(mono, sample_rate)
    total = float(np.sum(power))
    if total <= EPSILON:
        return {
            "rolloff_95_hz": 0.0,
            "rolloff_99_hz": 0.0,
            "energy_ratio_above_15khz": 0.0,
            "energy_ratio_above_18khz": 0.0,
            "energy_ratio_above_20khz": 0.0,
        }
    cumulative = np.cumsum(power)

    def rolloff(fraction: float) -> float:
        index = min(int(np.searchsorted(cumulative, total * fraction)), power.size - 1)
        return float(frequencies[index])

    def ratio_above(frequency: float) -> float | None:
        if frequencies[-1] < frequency:
            return None
        return float(np.sum(power[frequencies >= frequency]) / total)

    return {
        "rolloff_95_hz": rolloff(0.95),
        "rolloff_99_hz": rolloff(0.99),
        "energy_ratio_above_15khz": ratio_above(15000.0),
        "energy_ratio_above_18khz": ratio_above(18000.0),
        "energy_ratio_above_20khz": ratio_above(20000.0),
    }


class ProductionQualityMetric:
    """Measure loudness and production evidence without aesthetic thresholds."""

    name = "production_quality"

    def __init__(self, *, runtime_factory: RuntimeFactory = _create_runtime) -> None:
        self._runtime_factory = runtime_factory
        self._runtime: ProductionRuntime | None = None

    def _runtime_instance(self) -> ProductionRuntime:
        if self._runtime is None:
            self._runtime = self._runtime_factory()
        return self._runtime

    def evaluate(
        self,
        entry: ManifestEntry,
        candidate: AudioData,
        reference: AudioData | None,
        config: AnalysisConfig,
    ) -> MetricOutput:
        del entry, reference, config
        if candidate.frames == 0:
            raise ValueError("production quality cannot analyze empty audio")
        runtime = self._runtime_instance()
        samples = candidate.samples
        mono = samples.mean(axis=1)

        integrated_lufs: float | None = None
        loudness_range_lu: float | None = None
        loudness_reason: str | None = None
        if candidate.duration_seconds < 0.4:
            loudness_reason = "audio is shorter than the 400 ms BS.1770 gating block"
        else:
            integrated_lufs = _finite_or_none(
                runtime.integrated_loudness(samples, candidate.sample_rate)
            )
            if integrated_lufs is None:
                loudness_reason = "signal remained below the BS.1770 loudness gate"
            try:
                loudness_range_lu = _finite_or_none(
                    runtime.loudness_range(samples, candidate.sample_rate)
                )
            except ValueError:
                # EBU LRA requires enough short-term blocks; short songs legitimately
                # have no stable LRA rather than a failed evaluation.
                loudness_range_lu = None

        sample_peak = float(np.max(np.abs(samples)))
        oversampled = runtime.oversample(samples, TRUE_PEAK_OVERSAMPLE)
        estimated_true_peak = float(np.max(np.abs(oversampled)))
        rms = float(np.sqrt(np.mean(np.square(samples))))
        block_levels = _block_rms_levels(samples, candidate.sample_rate)
        block_summary = _percentiles(block_levels)
        block_summary["p90_minus_p10_db"] = block_summary["p90"] - block_summary["p10"]

        deltas = np.max(np.abs(np.diff(samples, axis=0)), axis=1)
        median_delta = float(np.median(deltas)) if deltas.size else 0.0
        mad = float(np.median(np.abs(deltas - median_delta))) if deltas.size else 0.0
        click_threshold = max(
            CLICK_ABSOLUTE_DELTA,
            median_delta + CLICK_MAD_MULTIPLIER * 1.4826 * mad,
        )
        click_indices = np.flatnonzero(deltas >= click_threshold)

        channel_true_peaks = []
        for channel in range(candidate.channels):
            peak = float(np.max(np.abs(oversampled[:, channel])))
            channel_true_peaks.append(
                {"channel": channel, "estimated_true_peak_dbtp": _db(peak)}
            )

        return MetricOutput(
            metrics={
                "implementation": {
                    "pyloudnorm_version": runtime.pyloudnorm_version,
                    "scipy_version": runtime.scipy_version,
                    "integrated_loudness": "ITU-R BS.1770-4 via pyloudnorm",
                    "loudness_range": "EBU Tech 3342 via pyloudnorm",
                    "true_peak": (
                        "4x scipy polyphase estimate; diagnostic, not a certified meter"
                    ),
                },
                "loudness": {
                    "integrated_lufs": integrated_lufs,
                    "loudness_range_lu": loudness_range_lu,
                    "loudness_range_stable": (
                        loudness_range_lu is not None
                        and candidate.duration_seconds >= 60.0
                    ),
                    "loudness_range_note": (
                        None
                        if candidate.duration_seconds >= 60.0
                        else "EBU guidance treats LRA as unstable before 60 seconds"
                    ),
                    "unavailable_reason": loudness_reason,
                    "sample_peak_dbfs": _db(sample_peak),
                    "estimated_true_peak_dbtp": _db(estimated_true_peak),
                    "crest_factor_db": _db(sample_peak) - _db(rms),
                    "block_rms_dbfs": block_summary,
                    "channel_true_peaks": channel_true_peaks,
                },
                "spectral": _spectral_evidence(runtime, mono, candidate.sample_rate),
                "impulsive_discontinuities": {
                    "method": "absolute first difference with robust adaptive floor",
                    "absolute_delta_floor": CLICK_ABSOLUTE_DELTA,
                    "mad_multiplier": CLICK_MAD_MULTIPLIER,
                    "effective_delta_threshold": click_threshold,
                    "maximum_absolute_delta": (
                        float(np.max(deltas)) if deltas.size else 0.0
                    ),
                    "candidate_count": int(click_indices.size),
                    "candidate_rate_per_second": (
                        click_indices.size / candidate.duration_seconds
                    ),
                    "first_candidate_seconds": [
                        float(index / candidate.sample_rate) for index in click_indices[:20]
                    ],
                },
            }
        )
