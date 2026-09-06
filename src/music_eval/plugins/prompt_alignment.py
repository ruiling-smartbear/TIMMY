from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from music_eval.audio import AudioData
from music_eval.models import ManifestEntry
from music_eval.plugins.base import MetricOutput


class PromptAlignmentBackend(Protocol):
    def score(
        self,
        audio_windows: Sequence[NDArray[np.float32]],
        sample_rate: int,
        texts: Sequence[str],
    ) -> NDArray[np.float64]: ...


def _score_summary(
    values: NDArray[np.float64], bounds: Sequence[tuple[int, int]], sample_rate: int
) -> dict[str, float]:
    worst_index = int(np.argmin(values))
    weights = np.asarray([end - start for start, end in bounds], dtype=np.float64)
    return {
        "mean": float(np.average(values, weights=weights)),
        "p10": float(np.percentile(values, 10)),
        "minimum": float(values[worst_index]),
        "worst_window_start_seconds": bounds[worst_index][0] / sample_rate,
    }


def _ranking(scores: NDArray[np.float64]) -> tuple[int, float | None, float | None]:
    positive = float(scores[0])
    if scores.size == 1:
        return 1, None, None
    best_negative = float(np.max(scores[1:]))
    rank = 1 + int(np.count_nonzero(scores[1:] > positive))
    return rank, best_negative, positive - best_negative


def evaluate_prompt_alignment(
    *,
    label: str,
    entry: ManifestEntry,
    candidate: AudioData,
    backend: PromptAlignmentBackend,
    window_seconds: float,
    batch_windows: int,
    metadata: dict[str, object],
) -> MetricOutput:
    """Apply a text/audio backend while preserving window-level evidence."""
    if entry.prompt is None or not entry.prompt.strip():
        raise ValueError(f"{label} alignment requires a nonempty manifest prompt")
    if candidate.frames == 0:
        raise ValueError(f"{label} alignment cannot score empty audio")

    texts = (entry.prompt.strip(), *entry.negative_prompts)
    window_frames = max(1, round(window_seconds * candidate.sample_rate))
    windows: list[NDArray[np.float32]] = []
    bounds: list[tuple[int, int]] = []
    for start in range(0, candidate.frames, window_frames):
        end = min(start + window_frames, candidate.frames)
        windows.append(
            np.asarray(
                candidate.samples[start:end].mean(axis=1),
                dtype=np.float32,
                order="C",
            )
        )
        bounds.append((start, end))

    score_batches = []
    for start in range(0, len(windows), batch_windows):
        score_batches.append(
            backend.score(
                windows[start : start + batch_windows],
                candidate.sample_rate,
                texts,
            )
        )
    scores = np.concatenate(score_batches, axis=0)
    expected_shape = (len(windows), len(texts))
    if scores.shape != expected_shape:
        raise ValueError(
            f"{label} backend returned shape {scores.shape}, expected {expected_shape}"
        )
    if not np.isfinite(scores).all():
        raise ValueError(f"{label} backend returned non-finite similarities")

    window_results = []
    margins = []
    for (start, end), row in zip(bounds, scores, strict=True):
        rank, best_negative, margin = _ranking(row)
        if margin is not None:
            margins.append(margin)
        window_results.append(
            {
                "start_seconds": start / candidate.sample_rate,
                "end_seconds": end / candidate.sample_rate,
                "positive_similarity": float(row[0]),
                "negative_similarities": [
                    {"text": text, "similarity": float(value)}
                    for text, value in zip(texts[1:], row[1:], strict=True)
                ],
                "best_negative_similarity": best_negative,
                "margin": margin,
                "positive_rank": rank,
            }
        )

    weights = np.asarray([end - start for start, end in bounds], dtype=np.float64)
    track_similarities = np.average(scores, axis=0, weights=weights)
    track_rank, best_negative, track_margin = _ranking(track_similarities)
    summary = {
        "positive_similarity": _score_summary(
            scores[:, 0], bounds, candidate.sample_rate
        )
    }
    if margins:
        summary["margin"] = _score_summary(
            np.asarray(margins, dtype=np.float64), bounds, candidate.sample_rate
        )

    return MetricOutput(
        metrics={
            **metadata,
            "prompt": texts[0],
            "negative_prompts": list(texts[1:]),
            "track": {
                "positive_similarity": float(track_similarities[0]),
                "negative_similarities": [
                    {"text": text, "similarity": float(value)}
                    for text, value in zip(
                        texts[1:], track_similarities[1:], strict=True
                    )
                ],
                "best_negative_similarity": best_negative,
                "margin": track_margin,
                "positive_rank": track_rank,
                "candidate_count": len(texts),
                "positive_top1": track_rank == 1,
            },
            "summary": summary,
            "windows": window_results,
        }
    )
