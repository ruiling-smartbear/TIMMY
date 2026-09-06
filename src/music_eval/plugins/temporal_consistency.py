from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from music_eval.audio import AudioData
from music_eval.metrics import dbfs
from music_eval.models import ManifestEntry
from music_eval.plugins.base import AnalysisConfig, MetricOutput

WINDOW_SECONDS = 2.0
HOP_SECONDS = 1.0
NONLOCAL_SECONDS = 8.0
PROFILE_BINS = 64
# Band power more than this far below the window's loudest band is clamped, so
# the profile describes spectral shape rather than the numerical noise floor.
PROFILE_RANGE_DB = 80.0
NEAR_DUPLICATE_SIMILARITY = 0.995
EPSILON = 1e-12


def _dbfs(rms: float) -> float:
    # Same floor as the integrity metric, so digital silence reads -120 dBFS everywhere.
    return dbfs(rms)


def _window_bounds(frames: int, window_frames: int, hop_frames: int) -> list[tuple[int, int]]:
    if frames <= window_frames:
        return [(0, frames)]
    starts = list(range(0, frames - window_frames + 1, hop_frames))
    final_start = frames - window_frames
    if starts[-1] != final_start:
        starts.append(final_start)
    return [(start, start + window_frames) for start in starts]


def _cosine(left: NDArray[np.float64], right: NDArray[np.float64]) -> float | None:
    # An all-zero profile (silent window) has no direction, so similarity is
    # undefined rather than zero; consumers skip None instead of counting drift.
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= EPSILON:
        return None
    return float(np.dot(left, right) / denominator)


def _spectral_features(
    samples: NDArray[np.float64], sample_rate: int, taper: NDArray[np.float64]
) -> tuple[dict[str, float], NDArray[np.float64]]:
    rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
    if samples.size < 2:
        return (
            {
                "rms_dbfs": _dbfs(rms),
                "spectral_centroid_hz": 0.0,
                "spectral_bandwidth_hz": 0.0,
                "spectral_rolloff_95_hz": 0.0,
                "spectral_flatness": 0.0,
                "zero_crossing_rate": 0.0,
            },
            np.zeros(PROFILE_BINS, dtype=np.float64),
        )

    windowed = samples * taper
    power = np.square(np.abs(np.fft.rfft(windowed)))
    frequencies = np.fft.rfftfreq(samples.size, d=1.0 / sample_rate)
    total_power = float(np.sum(power))
    if total_power <= EPSILON:
        centroid = bandwidth = rolloff = flatness = 0.0
        profile = np.zeros(PROFILE_BINS, dtype=np.float64)
    else:
        centroid = float(np.sum(frequencies * power) / total_power)
        bandwidth = float(
            np.sqrt(np.sum(np.square(frequencies - centroid) * power) / total_power)
        )
        cumulative = np.cumsum(power)
        rolloff_index = min(
            int(np.searchsorted(cumulative, total_power * 0.95)), frequencies.size - 1
        )
        rolloff = float(frequencies[rolloff_index])
        positive_power = power[1:] if power.size > 1 else power
        flatness = float(
            np.exp(np.mean(np.log(positive_power + EPSILON)))
            / (np.mean(positive_power) + EPSILON)
        )

        # Mean power in PROFILE_BINS log-spaced bands between 20 Hz and 20 kHz. Averaging
        # a band is a far lower-variance estimate than sampling single FFT bins.
        upper_hz = min(float(sample_rate) / 2.0, 20000.0)
        band_edges = np.geomspace(20.0, max(20.0, upper_hz), PROFILE_BINS + 1)
        edges = np.searchsorted(frequencies, band_edges)
        band_power = np.array(
            [
                power[low:high].mean() if high > low else 0.0
                for low, high in zip(edges[:-1], edges[1:], strict=True)
            ]
        )
        floor = float(band_power.max()) * 10.0 ** (-PROFILE_RANGE_DB / 10.0)
        profile = np.log10(np.maximum(band_power, floor) + EPSILON)
        profile -= float(np.mean(profile))
        norm = float(np.linalg.norm(profile))
        profile = profile / norm if norm > EPSILON else np.zeros(PROFILE_BINS, dtype=np.float64)

    signs = np.signbit(samples)
    zero_crossing_rate = float(np.mean(signs[1:] != signs[:-1]))
    return (
        {
            "rms_dbfs": _dbfs(rms),
            "spectral_centroid_hz": centroid,
            "spectral_bandwidth_hz": bandwidth,
            "spectral_rolloff_95_hz": rolloff,
            "spectral_flatness": flatness,
            "zero_crossing_rate": zero_crossing_rate,
        },
        profile,
    )


def _distribution(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "p05": float(np.percentile(array, 5)),
        "p95": float(np.percentile(array, 95)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


class TemporalConsistencyMetric:
    """Expose time-local spectral changes without assigning aesthetic policy."""

    name = "temporal_consistency"

    def __init__(
        self,
        *,
        window_seconds: float = WINDOW_SECONDS,
        hop_seconds: float = HOP_SECONDS,
        nonlocal_seconds: float = NONLOCAL_SECONDS,
        near_duplicate_similarity: float = NEAR_DUPLICATE_SIMILARITY,
    ) -> None:
        values = (window_seconds, hop_seconds, nonlocal_seconds)
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("temporal analysis durations must be positive and finite")
        if not 0.0 <= near_duplicate_similarity <= 1.0:
            raise ValueError("near_duplicate_similarity must be between 0 and 1")
        self._window_seconds = window_seconds
        self._hop_seconds = hop_seconds
        self._nonlocal_seconds = nonlocal_seconds
        self._near_duplicate_similarity = near_duplicate_similarity

    def evaluate(
        self,
        entry: ManifestEntry,
        candidate: AudioData,
        reference: AudioData | None,
        config: AnalysisConfig,
    ) -> MetricOutput:
        del entry, reference, config
        if candidate.frames == 0:
            raise ValueError("temporal consistency cannot analyze empty audio")

        mono = candidate.samples.mean(axis=1)
        window_frames = max(1, round(self._window_seconds * candidate.sample_rate))
        hop_frames = max(1, round(self._hop_seconds * candidate.sample_rate))
        bounds = _window_bounds(candidate.frames, window_frames, hop_frames)

        feature_rows = []
        profiles = []
        taper = np.hanning(bounds[0][1] - bounds[0][0])  # every window has the same length
        for start, end in bounds:
            features, profile = _spectral_features(mono[start:end], candidate.sample_rate, taper)
            feature_rows.append(
                {
                    "start_seconds": start / candidate.sample_rate,
                    "end_seconds": end / candidate.sample_rate,
                    **features,
                }
            )
            profiles.append(profile)

        adjacent_rms_jumps = []
        adjacent_centroid_jumps = []
        adjacent_similarities = []
        transitions = []
        for index in range(1, len(feature_rows)):
            rms_jump = abs(feature_rows[index]["rms_dbfs"] - feature_rows[index - 1]["rms_dbfs"])
            centroid_jump = abs(
                feature_rows[index]["spectral_centroid_hz"]
                - feature_rows[index - 1]["spectral_centroid_hz"]
            )
            similarity = _cosine(profiles[index - 1], profiles[index])
            adjacent_rms_jumps.append(rms_jump)
            adjacent_centroid_jumps.append(centroid_jump)
            if similarity is not None:
                adjacent_similarities.append(similarity)
            transitions.append(
                {
                    "at_seconds": feature_rows[index]["start_seconds"],
                    "rms_jump_db": rms_jump,
                    "centroid_jump_hz": centroid_jump,
                    "spectral_similarity": similarity,
                }
            )

        minimum_separation_frames = round(self._nonlocal_seconds * candidate.sample_rate)
        starts = np.array([start for start, _ in bounds])
        profile_matrix = np.vstack(profiles)
        # Profiles are unit-norm or all-zero, so the Gram matrix holds the cosines.
        # A pair with a zero profile has no defined similarity (see _cosine) and is
        # left out, the same way the adjacent-window loop above skips it.
        similarities = profile_matrix @ profile_matrix.T
        norms = np.linalg.norm(profile_matrix, axis=1)
        pair_mask = (
            np.triu(starts[None, :] - starts[:, None] >= minimum_separation_frames, k=1)
            & (np.outer(norms, norms) > EPSILON)
        )
        all_nonlocal_similarities: list[float] = similarities[pair_mask].tolist()
        nonlocal_mask = pair_mask | pair_mask.T
        per_window_nonlocal_max = [
            float(np.max(row[mask])) if mask.any() else None
            for row, mask in zip(similarities, nonlocal_mask, strict=True)
        ]

        comparable = [value for value in per_window_nonlocal_max if value is not None]
        duplicate_count = sum(
            value >= self._near_duplicate_similarity for value in comparable
        )
        first = feature_rows[0]
        last = feature_rows[-1]
        tail_frames = max(1, round(0.25 * candidate.sample_rate))
        tail = mono[-tail_frames:]
        previous_tail = mono[-2 * tail_frames : -tail_frames]
        tail_rms = _dbfs(float(np.sqrt(np.mean(np.square(tail)))))
        previous_tail_rms = (
            _dbfs(float(np.sqrt(np.mean(np.square(previous_tail)))))
            if previous_tail.size
            else tail_rms
        )

        return MetricOutput(
            metrics={
                "preprocessing": {
                    "channels": "mono mean",
                    "window_seconds": self._window_seconds,
                    "hop_seconds": self._hop_seconds,
                    "profile_bins": PROFILE_BINS,
                    "profile_range_db": PROFILE_RANGE_DB,
                    "nonlocal_minimum_separation_seconds": self._nonlocal_seconds,
                    "near_duplicate_similarity": self._near_duplicate_similarity,
                },
                "windows": feature_rows,
                "transitions": transitions,
                "transition_summary": {
                    "rms_jump_db": _distribution(adjacent_rms_jumps),
                    "centroid_jump_hz": _distribution(adjacent_centroid_jumps),
                    "spectral_similarity": _distribution(adjacent_similarities),
                },
                "first_to_last": {
                    "rms_delta_db": last["rms_dbfs"] - first["rms_dbfs"],
                    "centroid_delta_hz": (
                        last["spectral_centroid_hz"] - first["spectral_centroid_hz"]
                    ),
                    "spectral_similarity": _cosine(profiles[0], profiles[-1]),
                },
                "nonlocal_repetition": {
                    "pair_count": len(all_nonlocal_similarities),
                    "similarity": _distribution(all_nonlocal_similarities),
                    "comparable_window_count": len(comparable),
                    "near_duplicate_window_ratio": (
                        duplicate_count / len(comparable) if comparable else None
                    ),
                },
                "ending": {
                    "tail_seconds": min(0.25, candidate.duration_seconds),
                    "tail_rms_dbfs": tail_rms,
                    "tail_to_previous_rms_delta_db": tail_rms - previous_tail_rms,
                    "last_sample_absolute": float(abs(mono[-1])),
                },
            }
        )
