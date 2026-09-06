from __future__ import annotations

from pathlib import Path

import numpy as np

from music_eval.audio import AudioData
from music_eval.models import ManifestEntry
from music_eval.plugins.base import AnalysisConfig
from music_eval.plugins.production_quality import ProductionQualityMetric


class FakeRuntime:
    pyloudnorm_version = "test"
    scipy_version = "test"

    def __init__(self, integrated=-14.0, loudness_range=6.0):
        self.integrated = integrated
        self.range = loudness_range
        self.loudness_calls = 0

    def integrated_loudness(self, samples, sample_rate):
        self.loudness_calls += 1
        return self.integrated

    def loudness_range(self, samples, sample_rate):
        return self.range

    def oversample(self, samples, factor):
        return np.repeat(samples, factor, axis=0)

    def welch(self, samples, sample_rate):
        frequencies = np.asarray([0, 10000, 16000, 19000, 22000], dtype=np.float64)
        power = np.asarray([8, 1, 0.5, 0.25, 0.25], dtype=np.float64)
        return frequencies, power


def _audio(samples: np.ndarray, sample_rate: int = 48000) -> AudioData:
    shaped = np.asarray(samples, dtype=np.float64)
    if shaped.ndim == 1:
        shaped = shaped[:, None]
    shaped.setflags(write=False)
    return AudioData(shaped, sample_rate, shaped.shape[1], 2)


def _evaluate(samples: np.ndarray, runtime: FakeRuntime, sample_rate: int = 48000):
    return ProductionQualityMetric(runtime_factory=lambda: runtime).evaluate(
        ManifestEntry(id="song", audio=Path("song.wav")),
        _audio(samples, sample_rate),
        None,
        AnalysisConfig(),
    ).metrics


def test_production_quality_reports_loudness_dynamics_and_bandwidth():
    time = np.arange(48000) / 48000
    samples = 0.25 * np.sin(2 * np.pi * 440 * time)

    metrics = _evaluate(samples, FakeRuntime())

    assert metrics["loudness"]["integrated_lufs"] == -14.0
    assert metrics["loudness"]["loudness_range_lu"] == 6.0
    assert metrics["loudness"]["loudness_range_stable"] is False
    assert "before 60 seconds" in metrics["loudness"]["loudness_range_note"]
    assert metrics["loudness"]["estimated_true_peak_dbtp"] == metrics["loudness"][
        "sample_peak_dbfs"
    ]
    assert metrics["loudness"]["block_rms_dbfs"]["p90_minus_p10_db"] < 1e-9
    assert metrics["spectral"]["energy_ratio_above_15khz"] == 0.1
    assert "certified meter" in metrics["implementation"]["true_peak"]


def test_production_quality_exposes_impulsive_discontinuity():
    samples = np.zeros(48000)
    samples[24000] = 1.0
    samples[24001] = -1.0

    metrics = _evaluate(samples, FakeRuntime())

    impulses = metrics["impulsive_discontinuities"]
    assert impulses["candidate_count"] >= 2
    assert impulses["maximum_absolute_delta"] == 2.0
    assert impulses["first_candidate_seconds"][0] < 0.51


def test_short_audio_marks_loudness_unavailable_without_calling_meter():
    runtime = FakeRuntime()

    metrics = _evaluate(np.zeros(100), runtime)

    assert metrics["loudness"]["integrated_lufs"] is None
    assert "shorter than" in metrics["loudness"]["unavailable_reason"]
    assert runtime.loudness_calls == 0


def test_below_gate_loudness_is_json_safe_none():
    runtime = FakeRuntime(integrated=-float("inf"))

    metrics = _evaluate(np.zeros(48000), runtime)

    assert metrics["loudness"]["integrated_lufs"] is None
    assert "below" in metrics["loudness"]["unavailable_reason"]


def test_low_sample_rate_marks_unavailable_high_frequency_ratios():
    runtime = FakeRuntime()
    runtime.welch = lambda samples, sample_rate: (
        np.asarray([0.0, 4000.0]),
        np.asarray([1.0, 1.0]),
    )

    metrics = _evaluate(np.zeros(8000), runtime, sample_rate=8000)

    assert metrics["spectral"]["energy_ratio_above_15khz"] is None


def test_channel_true_peaks_preserve_stereo_imbalance():
    left = np.full(48000, 0.5)
    right = np.full(48000, 0.25)

    metrics = _evaluate(np.column_stack((left, right)), FakeRuntime())

    channels = metrics["loudness"]["channel_true_peaks"]
    assert channels[0]["estimated_true_peak_dbtp"] > channels[1][
        "estimated_true_peak_dbtp"
    ]
