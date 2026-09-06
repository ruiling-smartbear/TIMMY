from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from music_eval.audio import AudioData
from music_eval.models import ManifestEntry
from music_eval.plugins.base import AnalysisConfig
from music_eval.plugins.temporal_consistency import (
    NEAR_DUPLICATE_SIMILARITY,
    TemporalConsistencyMetric,
    _cosine,
    _distribution,
    _spectral_features,
    _window_bounds,
)


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


def test_silent_opening_window_has_undefined_similarity_instead_of_maximum_drift():
    samples = np.concatenate((np.zeros(2 * 4000), _tone(220, 10, 4000)))

    metrics = _evaluate(samples)

    assert metrics["first_to_last"]["spectral_similarity"] is None
    assert metrics["transitions"][0]["spectral_similarity"] is None
    assert all(
        transition["spectral_similarity"] is not None
        for transition in metrics["transitions"][1:]
    )
    assert metrics["transition_summary"]["spectral_similarity"]["minimum"] > 0.5
    assert metrics["nonlocal_repetition"]["similarity"]["minimum"] > 0.5
    # Pairs (0,8), (0,9), (0,10) touch the silent window and are skipped.
    assert metrics["nonlocal_repetition"]["pair_count"] == 3
    assert metrics["nonlocal_repetition"]["comparable_window_count"] == 4

def _oracle_nonlocal_repetition(
    profiles: list[np.ndarray],
    bounds: list[tuple[int, int]],
    minimum_separation_frames: int,
    near_duplicate_similarity: float,
) -> dict:
    """Verbatim copy of the pre-vectorization O(n^2) pair loop, kept as the oracle."""
    per_window_nonlocal_max: list[float | None] = [None] * len(bounds)
    all_nonlocal_similarities = []
    for left in range(len(bounds)):
        for right in range(left + 1, len(bounds)):
            if bounds[right][0] - bounds[left][0] < minimum_separation_frames:
                continue
            similarity = _cosine(profiles[left], profiles[right])
            if similarity is None:
                continue
            all_nonlocal_similarities.append(similarity)
            current_left = per_window_nonlocal_max[left]
            current_right = per_window_nonlocal_max[right]
            per_window_nonlocal_max[left] = (
                similarity if current_left is None else max(current_left, similarity)
            )
            per_window_nonlocal_max[right] = (
                similarity if current_right is None else max(current_right, similarity)
            )
    comparable = [value for value in per_window_nonlocal_max if value is not None]
    duplicate_count = sum(value >= near_duplicate_similarity for value in comparable)
    return {
        "pair_count": len(all_nonlocal_similarities),
        "similarity": _distribution(all_nonlocal_similarities),
        "comparable_window_count": len(comparable),
        "near_duplicate_window_ratio": (
            duplicate_count / len(comparable) if comparable else None
        ),
    }


def _drifting_tone(frequency: float, seconds: float, period: float, sample_rate: int) -> np.ndarray:
    """A tone whose pitch wobbles slowly, so that windows are similar but not identical."""
    time = np.arange(round(seconds * sample_rate)) / sample_rate
    wobble = 0.02 * period / (2 * np.pi) * np.sin(2 * np.pi * time / period)
    return 0.25 * np.sin(2 * np.pi * frequency * (time + wobble))


@pytest.mark.parametrize("nonlocal_seconds", [8.0, 1e-9, 100.0])
def test_vectorized_nonlocal_search_matches_pair_loop_oracle(nonlocal_seconds):
    sample_rate = 4000
    motif = _drifting_tone(220, 10, 13, sample_rate) + _drifting_tone(330, 10, 7, sample_rate)
    bridge = _tone(660, 6, sample_rate) + _tone(990, 6, sample_rate)
    samples = np.concatenate((motif, bridge, motif))
    metric = TemporalConsistencyMetric(nonlocal_seconds=nonlocal_seconds)

    actual = metric.evaluate(
        ManifestEntry(id="song", audio=Path("song.wav")),
        _audio(samples, sample_rate),
        None,
        AnalysisConfig(),
    ).metrics["nonlocal_repetition"]

    bounds = _window_bounds(samples.size, 2 * sample_rate, sample_rate)
    taper = np.hanning(2 * sample_rate)
    profiles = [
        _spectral_features(samples[start:end], sample_rate, taper)[1] for start, end in bounds
    ]
    expected = _oracle_nonlocal_repetition(
        profiles, bounds, round(nonlocal_seconds * sample_rate), NEAR_DUPLICATE_SIMILARITY
    )
    assert actual["pair_count"] == expected["pair_count"]
    assert actual["comparable_window_count"] == expected["comparable_window_count"]
    assert actual["near_duplicate_window_ratio"] == expected["near_duplicate_window_ratio"]
    if expected["similarity"] is None:
        assert actual["similarity"] is None
    else:
        assert actual["similarity"].keys() == expected["similarity"].keys()
        for key, value in expected["similarity"].items():
            assert actual["similarity"][key] == pytest.approx(value, abs=1e-12), key
    if nonlocal_seconds == 8.0:
        # The motif windows repeat after the bridge; the bridge windows do not.
        assert 0.0 < actual["near_duplicate_window_ratio"] < 1.0
        assert actual["similarity"]["minimum"] < 0.5 < actual["similarity"]["maximum"]


def test_silent_window_level_uses_the_shared_dbfs_floor():
    """Digital silence reads -120 dBFS here, as in the integrity metric."""
    samples = np.concatenate((np.zeros(2 * 4000), _tone(220, 4, 4000)))

    metrics = _evaluate(samples)

    assert metrics["windows"][0]["rms_dbfs"] == -120.0
