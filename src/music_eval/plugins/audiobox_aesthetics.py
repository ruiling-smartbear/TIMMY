from __future__ import annotations

import math
from collections.abc import Callable
from importlib import import_module
from typing import Any

import numpy as np
from numpy.typing import NDArray

from music_eval.audio import AudioData
from music_eval.models import ManifestEntry
from music_eval.plugins.base import AnalysisConfig, MetricOutput
from music_eval.text import package_version

AXES = ("CE", "CU", "PC", "PQ")
CHECKPOINT = "facebook/audiobox-aesthetics"
OFFICIAL_WINDOW_SECONDS = 10.0
DEFAULT_BATCH_WINDOWS = 8

PredictorFactory = Callable[[], Any]
TensorFactory = Callable[[NDArray[np.float32]], Any]


def _create_predictor() -> Any:
    try:
        module = import_module("audiobox_aesthetics.infer")
    except ImportError as error:
        raise RuntimeError(
            "Audiobox Aesthetics is optional; install music-eval[audiobox]"
        ) from error
    return module.initialize_predictor()


def _create_tensor(samples: NDArray[np.float32]) -> Any:
    try:
        torch = import_module("torch")
    except ImportError as error:
        raise RuntimeError(
            "Audiobox Aesthetics requires PyTorch; install music-eval[audiobox]"
        ) from error
    return torch.from_numpy(samples)



def _validated_scores(prediction: object) -> dict[str, float]:
    if not isinstance(prediction, dict):
        raise ValueError("Audiobox predictor returned a non-object result")
    scores: dict[str, float] = {}
    for axis in AXES:
        value = prediction.get(axis)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Audiobox predictor returned an invalid {axis} score")
        score = float(value)
        if not math.isfinite(score):
            raise ValueError(f"Audiobox predictor returned a non-finite {axis} score")
        scores[axis] = score
    return scores


class AudioboxAestheticsMetric:
    """Run Meta's no-reference aesthetic model with visible window evidence."""

    name = "audiobox_aesthetics"

    def __init__(
        self,
        *,
        predictor_factory: PredictorFactory = _create_predictor,
        tensor_factory: TensorFactory = _create_tensor,
        window_seconds: float = OFFICIAL_WINDOW_SECONDS,
        batch_windows: int = DEFAULT_BATCH_WINDOWS,
    ) -> None:
        if not math.isfinite(window_seconds) or window_seconds <= 0:
            raise ValueError("Audiobox window_seconds must be positive and finite")
        if batch_windows <= 0:
            raise ValueError("Audiobox batch_windows must be positive")
        self._predictor_factory = predictor_factory
        self._tensor_factory = tensor_factory
        self._window_seconds = window_seconds
        self._batch_windows = batch_windows
        self._predictor: Any | None = None

    def _predictor_instance(self) -> Any:
        if self._predictor is None:
            self._predictor = self._predictor_factory()
        return self._predictor

    def evaluate(
        self,
        entry: ManifestEntry,
        candidate: AudioData,
        reference: AudioData | None,
        config: AnalysisConfig,
    ) -> MetricOutput:
        del entry, reference, config
        if candidate.frames == 0:
            raise ValueError("Audiobox Aesthetics cannot score empty audio")

        window_frames = max(1, round(self._window_seconds * candidate.sample_rate))
        model_inputs: list[dict[str, Any]] = []
        bounds: list[tuple[int, int]] = []
        for start in range(0, candidate.frames, window_frames):
            end = min(start + window_frames, candidate.frames)
            # The official predictor accepts channel-first tensors and performs
            # its documented mono conversion and 16 kHz resampling internally.
            channel_first = np.asarray(
                candidate.samples[start:end].T, dtype=np.float32, order="C"
            )
            model_inputs.append(
                {
                    "path": self._tensor_factory(channel_first),
                    "sample_rate": candidate.sample_rate,
                }
            )
            bounds.append((start, end))

        raw_predictions: list[object] = []
        predictor = self._predictor_instance()
        for start in range(0, len(model_inputs), self._batch_windows):
            batch_predictions = predictor.forward(
                model_inputs[start : start + self._batch_windows]
            )
            if not isinstance(batch_predictions, list):
                raise ValueError("Audiobox predictor returned a non-list result")
            raw_predictions.extend(batch_predictions)
        if len(raw_predictions) != len(bounds):
            raise ValueError(
                "Audiobox predictor returned a different number of results than windows"
            )
        predictions = [_validated_scores(item) for item in raw_predictions]

        windows = []
        for (start, end), scores in zip(bounds, predictions, strict=True):
            windows.append(
                {
                    "start_seconds": start / candidate.sample_rate,
                    "end_seconds": end / candidate.sample_rate,
                    "scores": scores,
                }
            )

        weights = np.asarray([end - start for start, end in bounds], dtype=np.float64)
        track_scores: dict[str, float] = {}
        summaries: dict[str, dict[str, float]] = {}
        for axis in AXES:
            values = np.asarray([item[axis] for item in predictions], dtype=np.float64)
            mean = float(np.average(values, weights=weights))
            worst_index = int(np.argmin(values))
            track_scores[axis] = mean
            summaries[axis] = {
                "mean": mean,
                "p10": float(np.percentile(values, 10)),
                "minimum": float(values[worst_index]),
                "worst_window_start_seconds": bounds[worst_index][0]
                / candidate.sample_rate,
            }

        return MetricOutput(
            metrics={
                "checkpoint": CHECKPOINT,
                "package_version": package_version("audiobox_aesthetics"),
                "preprocessing": {
                    "model_sample_rate_hz": 16000,
                    "channels": "mono",
                    "window_seconds": self._window_seconds,
                    "hop_seconds": self._window_seconds,
                    "inference_batch_windows": self._batch_windows,
                    "aggregation": "duration-weighted mean",
                },
                "track_scores": track_scores,
                "axis_summary": summaries,
                "windows": windows,
            }
        )
