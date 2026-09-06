from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from music_eval.audio import AudioData
from music_eval.models import ManifestEntry
from music_eval.plugins.base import AnalysisConfig, MetricOutput

CHECKPOINT = "laion/clap-htsat-unfused"
MODEL_SAMPLE_RATE = 48000
WINDOW_SECONDS = 10.0
DEFAULT_BATCH_WINDOWS = 8


class AlignmentBackend(Protocol):
    device_name: str

    def score(
        self,
        audio_windows: Sequence[NDArray[np.float32]],
        sample_rate: int,
        texts: Sequence[str],
    ) -> NDArray[np.float64]: ...


BackendFactory = Callable[[], AlignmentBackend]


def _package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unknown"


class TransformersClapBackend:
    """Thin lazy adapter over Hugging Face's CLAP implementation."""

    def __init__(self, checkpoint: str = CHECKPOINT) -> None:
        try:
            torch = import_module("torch")
            transformers = import_module("transformers")
            self._soxr = import_module("soxr")
        except ImportError as error:
            raise RuntimeError(
                "CLAP alignment is optional; install music-eval[clap]"
            ) from error

        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        self.device_name = device
        self._torch = torch
        self._processor = transformers.AutoProcessor.from_pretrained(checkpoint)
        self._model = transformers.AutoModel.from_pretrained(checkpoint)
        self._model.to(device)
        self._model.eval()

    def score(
        self,
        audio_windows: Sequence[NDArray[np.float32]],
        sample_rate: int,
        texts: Sequence[str],
    ) -> NDArray[np.float64]:
        resampled = [
            np.asarray(
                self._soxr.resample(
                    window,
                    sample_rate,
                    MODEL_SAMPLE_RATE,
                    quality="HQ",
                ),
                dtype=np.float32,
            )
            if sample_rate != MODEL_SAMPLE_RATE
            else window
            for window in audio_windows
        ]
        inputs = self._processor(
            text=list(texts),
            audio=resampled,
            sampling_rate=MODEL_SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        inputs = {name: value.to(self.device_name) for name, value in inputs.items()}
        with self._torch.inference_mode():
            output = self._model(**inputs)
            audio_embeddings = self._torch.nn.functional.normalize(
                output.audio_embeds.float(), dim=-1
            )
            text_embeddings = self._torch.nn.functional.normalize(
                output.text_embeds.float(), dim=-1
            )
            similarities = audio_embeddings @ text_embeddings.T
        return np.asarray(similarities.cpu().numpy(), dtype=np.float64)


def _create_backend() -> AlignmentBackend:
    return TransformersClapBackend()


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


class ClapAlignmentMetric:
    """Measure positive-prompt alignment against explicit hard negatives."""

    name = "clap_alignment"

    def __init__(
        self,
        *,
        backend_factory: BackendFactory = _create_backend,
        window_seconds: float = WINDOW_SECONDS,
        batch_windows: int = DEFAULT_BATCH_WINDOWS,
    ) -> None:
        if not math.isfinite(window_seconds) or window_seconds <= 0:
            raise ValueError("CLAP window_seconds must be positive and finite")
        if batch_windows <= 0:
            raise ValueError("CLAP batch_windows must be positive")
        self._backend_factory = backend_factory
        self._window_seconds = window_seconds
        self._batch_windows = batch_windows
        self._backend: AlignmentBackend | None = None

    def _backend_instance(self) -> AlignmentBackend:
        if self._backend is None:
            self._backend = self._backend_factory()
        return self._backend

    def evaluate(
        self,
        entry: ManifestEntry,
        candidate: AudioData,
        reference: AudioData | None,
        config: AnalysisConfig,
    ) -> MetricOutput:
        del reference, config
        if entry.prompt is None or not entry.prompt.strip():
            raise ValueError("CLAP alignment requires a nonempty manifest prompt")
        if candidate.frames == 0:
            raise ValueError("CLAP alignment cannot score empty audio")

        texts = (entry.prompt.strip(), *entry.negative_prompts)
        window_frames = max(1, round(self._window_seconds * candidate.sample_rate))
        windows: list[NDArray[np.float32]] = []
        bounds: list[tuple[int, int]] = []
        for start in range(0, candidate.frames, window_frames):
            end = min(start + window_frames, candidate.frames)
            mono = np.asarray(
                candidate.samples[start:end].mean(axis=1), dtype=np.float32, order="C"
            )
            windows.append(mono)
            bounds.append((start, end))

        backend = self._backend_instance()
        score_batches = []
        for start in range(0, len(windows), self._batch_windows):
            score_batches.append(
                backend.score(
                    windows[start : start + self._batch_windows],
                    candidate.sample_rate,
                    texts,
                )
            )
        scores = np.concatenate(score_batches, axis=0)
        expected_shape = (len(windows), len(texts))
        if scores.shape != expected_shape:
            raise ValueError(
                f"CLAP backend returned shape {scores.shape}, expected {expected_shape}"
            )
        if not np.isfinite(scores).all():
            raise ValueError("CLAP backend returned non-finite similarities")

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
                        {"text": text, "similarity": float(score)}
                        for text, score in zip(texts[1:], row[1:], strict=True)
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
                "checkpoint": CHECKPOINT,
                "transformers_version": _package_version("transformers"),
                "device": getattr(backend, "device_name", "unknown"),
                "preprocessing": {
                    "model_sample_rate_hz": MODEL_SAMPLE_RATE,
                    "channels": "mono",
                    "resampler": "soxr HQ",
                    "window_seconds": self._window_seconds,
                    "hop_seconds": self._window_seconds,
                    "inference_batch_windows": self._batch_windows,
                    "aggregation": "duration-weighted cosine similarity",
                },
                "prompt": texts[0],
                "negative_prompts": list(texts[1:]),
                "track": {
                    "positive_similarity": float(track_similarities[0]),
                    "negative_similarities": [
                        {"text": text, "similarity": float(score)}
                        for text, score in zip(
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
