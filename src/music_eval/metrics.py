from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from music_eval.models import Dropout


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


def detect_dropouts(
    samples: NDArray[np.float64],
    sample_rate: int,
    *,
    silence_dbfs: float,
    window_seconds: float,
    minimum_seconds: float,
) -> tuple[list[Dropout], float, float]:
    rms_values, window_frames = _mono_rms_windows(
        samples, sample_rate, window_seconds
    )
    silent = np.asarray([dbfs(float(value)) <= silence_dbfs for value in rms_values])
    dropouts: list[Dropout] = []
    start_index: int | None = None

    for index, is_silent in enumerate(np.append(silent, False)):
        if is_silent and start_index is None:
            start_index = index
        elif not is_silent and start_index is not None:
            start_frame = start_index * window_frames
            end_frame = min(index * window_frames, samples.shape[0])
            duration = (end_frame - start_frame) / sample_rate
            if duration + 1e-12 >= minimum_seconds:
                dropouts.append(
                    Dropout(
                        start_seconds=start_frame / sample_rate,
                        end_seconds=end_frame / sample_rate,
                        duration_seconds=duration,
                    )
                )
            start_index = None

    silent_ratio = float(np.mean(silent)) if silent.size else 1.0
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
