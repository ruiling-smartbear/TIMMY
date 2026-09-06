from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from importlib import import_module
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from music_eval.audio import AudioData
from music_eval.models import ManifestEntry
from music_eval.plugins.base import AnalysisConfig, MetricOutput
from music_eval.plugins.prompt_alignment import evaluate_prompt_alignment
from music_eval.text import package_version

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
        backend = self._backend_instance()
        return evaluate_prompt_alignment(
            label="CLAP",
            entry=entry,
            candidate=candidate,
            backend=backend,
            window_seconds=self._window_seconds,
            batch_windows=self._batch_windows,
            metadata={
                "checkpoint": CHECKPOINT,
                "transformers_version": package_version("transformers"),
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
            },
        )
