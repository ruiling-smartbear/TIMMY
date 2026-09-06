from __future__ import annotations

from pathlib import Path

import numpy as np

from music_eval.audio import AudioData
from music_eval.models import ManifestEntry
from music_eval.plugins.base import AnalysisConfig
from music_eval.plugins.temporal_consistency import TemporalConsistencyMetric


def _audio(samples: np.ndarray, sample_rate: int) -> AudioData:
    shaped = np.asarray(samples, dtype=np.float64).reshape(-1, 1)
    shaped.setflags(write=False)
    return AudioData(shaped, sample_rate, 1, 2)


def _tone(frequency: float, seconds: float, sample_rate: int) -> np.ndarray:
    time = np.arange(round(seconds * sample_rate)) / sample_rate
    return 0.25 * np.sin(2 * np.pi * frequency * time)


def _evaluate(samples: np.ndarray, sample_rate: int = 4000):
    return TemporalConsistencyMetric().evaluate(
        ManifestEntry(id="song", audio=Path("song.wav")),
        _audio(samples, sample_rate),
        None,
        AnalysisConfig(),
    ).metrics


def test_temporal_metric_keeps_stable_tone_as_visible_repetition_evidence():
    metrics = _evaluate(_tone(220, 12, 4000))

    assert len(metrics["windows"]) == 11
    assert metrics["first_to_last"]["spectral_similarity"] > 0.999
    assert metrics["nonlocal_repetition"]["near_duplicate_window_ratio"] == 1.0
    assert metrics["transition_summary"]["spectral_similarity"]["minimum"] > 0.999


def test_temporal_metric_exposes_an_abrupt_frequency_change():
    samples = np.concatenate((_tone(220, 6, 4000), _tone(1200, 6, 4000)))

    metrics = _evaluate(samples)

    assert metrics["first_to_last"]["spectral_similarity"] < 0.8
    assert metrics["transition_summary"]["centroid_jump_hz"]["maximum"] > 400
    assert metrics["transition_summary"]["spectral_similarity"]["minimum"] < 0.8


def test_temporal_metric_anchors_last_window_to_track_end():
    metrics = _evaluate(_tone(220, 3.25, 4000))

    assert metrics["windows"][-1]["end_seconds"] == 3.25
    assert metrics["ending"]["tail_seconds"] == 0.25


def test_short_audio_has_no_fake_transition_or_nonlocal_pair():
    metrics = _evaluate(_tone(220, 0.5, 4000))

    assert len(metrics["windows"]) == 1
    assert metrics["transition_summary"]["spectral_similarity"] is None
    assert metrics["nonlocal_repetition"]["pair_count"] == 0
    assert metrics["nonlocal_repetition"]["near_duplicate_window_ratio"] is None
