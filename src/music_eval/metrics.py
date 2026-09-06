from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from music_eval.models import Dropout

# Dropouts are located on this fine frame grid, then refined to sample accuracy.
DROPOUT_FRAME_SECONDS = 0.01


def dbfs(value: float, floor_db: float = -120.0) -> float:
    if value <= 0.0:
        return floor_db
    return max(floor_db, 20.0 * math.log10(value))


def _mono_rms_windows(
    samples: NDArray[np.float64], sample_rate: int, window_seconds: float
) -> tuple[NDArray[np.float64], int]:
    mono = np.sqrt(np.mean(np.square(samples), axis=1))
    window_frames = max(1, round(sample_rate * window_seconds))
    if mono.size == 0:
        return np.empty(0, dtype=np.float64), window_frames
    rms_values = []
    for start in range(0, mono.size, window_frames):
        window = mono[start : start + window_frames]
        rms_values.append(float(np.sqrt(np.mean(np.square(window)))))
    return np.asarray(rms_values), window_frames


def _silent_windows(
    samples: NDArray[np.float64],
    sample_rate: int,
    window_seconds: float,
    silence_dbfs: float,
) -> tuple[NDArray[np.bool_], int]:
    rms_values, window_frames = _mono_rms_windows(samples, sample_rate, window_seconds)
    silent = [dbfs(float(value)) <= silence_dbfs for value in rms_values]
    return np.asarray(silent, dtype=bool), window_frames


def _silent_runs(silent: NDArray[np.bool_]) -> list[tuple[int, int]]:
    """Run-length encode consecutive silent windows as [start, stop) index pairs."""
    padded = np.concatenate(([False], silent, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1]).reshape(-1, 2)
    return [(int(start), int(stop)) for start, stop in edges]


def _loud_sample_indices(
    samples: NDArray[np.float64], start: int, stop: int, amplitude: float
) -> NDArray[np.intp]:
    """Indices, relative to ``start``, whose max-abs across channels exceeds ``amplitude``."""
    return np.flatnonzero(np.max(np.abs(samples[start:stop]), axis=1) > amplitude)


def detect_dropouts(
    samples: NDArray[np.float64],
    sample_rate: int,
    *,
    silence_dbfs: float,
    window_seconds: float,
    minimum_seconds: float,
) -> tuple[list[Dropout], float, float]:
    # silent_window_ratio keeps the documented window grid. Dropouts are found on
    # DROPOUT_FRAME_SECONDS frames and each boundary is then moved to the exact
    # sample by inspecting the partial frame on either side of the silent run, so
    # a dropout that straddles a window boundary is neither missed nor shortened.
    silent_windows, _window_frames = _silent_windows(
        samples, sample_rate, window_seconds, silence_dbfs
    )
    silent_frames, frame_samples = _silent_windows(
        samples, sample_rate, DROPOUT_FRAME_SECONDS, silence_dbfs
    )
    amplitude = 10.0 ** (silence_dbfs / 20.0)
    dropouts: list[Dropout] = []

    for first_frame, stop_frame in _silent_runs(silent_frames):
        start = first_frame * frame_samples
        end = min(stop_frame * frame_samples, samples.shape[0])
        if start > 0:
            loud = _loud_sample_indices(samples, start - frame_samples, start, amplitude)
            if loud.size:
                start = start - frame_samples + int(loud[-1]) + 1
        if end < samples.shape[0]:
            loud = _loud_sample_indices(samples, end, end + frame_samples, amplitude)
            if loud.size:
                end += int(loud[0])
        duration = (end - start) / sample_rate
        if duration + 1e-12 >= minimum_seconds:
            dropouts.append(
                Dropout(
                    start_seconds=start / sample_rate,
                    end_seconds=end / sample_rate,
                    duration_seconds=duration,
                )
            )

    silent_ratio = float(np.mean(silent_windows)) if silent_windows.size else 1.0
    longest = max((dropout.duration_seconds for dropout in dropouts), default=0.0)
    return dropouts, silent_ratio, longest


def summarize_signal(
    samples: NDArray[np.float64],
    sample_rate: int,
    *,
    silence_dbfs: float,
    window_seconds: float,
    minimum_dropout_seconds: float,
    clip_threshold: float,
) -> tuple[dict[str, object], list[Dropout]]:
    absolute = np.abs(samples)
    rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
    peak = float(np.max(absolute)) if samples.size else 0.0
    channel_rms = (
        np.sqrt(np.mean(np.square(samples), axis=0))
        if samples.size
        else np.zeros(samples.shape[1])
    )
    dropouts, silent_ratio, longest_dropout = detect_dropouts(
        samples,
        sample_rate,
        silence_dbfs=silence_dbfs,
        window_seconds=window_seconds,
        minimum_seconds=minimum_dropout_seconds,
    )

    metrics: dict[str, object] = {
        "rms_dbfs": dbfs(rms),
        "peak_dbfs": dbfs(peak),
        "dc_offset": [float(value) for value in np.mean(samples, axis=0)]
        if samples.size
        else [],
        "clipped_sample_ratio": float(np.mean(absolute >= clip_threshold))
        if samples.size
        else 0.0,
        "silent_window_ratio": silent_ratio,
        "longest_dropout_seconds": longest_dropout,
        "channel_rms_dbfs": [dbfs(float(value)) for value in channel_rms],
    }
    if samples.shape[1] == 2 and samples.shape[0] > 1:
        left_std = float(np.std(samples[:, 0]))
        right_std = float(np.std(samples[:, 1]))
        metrics["stereo_correlation"] = (
            float(np.corrcoef(samples[:, 0], samples[:, 1])[0, 1])
            if left_std > 0.0 and right_std > 0.0
            else None
        )
    return metrics, dropouts
