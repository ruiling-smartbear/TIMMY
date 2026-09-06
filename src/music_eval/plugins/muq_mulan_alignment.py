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
from music_eval.plugins.prompt_alignment import evaluate_prompt_alignment

CHECKPOINT = "OpenMuQ/MuQ-MuLan-large"
MODEL_SAMPLE_RATE = 24000
WINDOW_SECONDS = 10.0
DEFAULT_BATCH_WINDOWS = 4


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


class MuQMuLanBackend:
    """Lazy adapter over Tencent AI Lab's official MuQ-MuLan package."""

    def __init__(self, checkpoint: str = CHECKPOINT) -> None:
        try:
            torch = import_module("torch")
            muq = import_module("muq")
            self._soxr = import_module("soxr")
        except ImportError as error:
            raise RuntimeError(
                "MuQ-MuLan alignment is optional; install music-eval[muq]"
            ) from error

        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        self.device_name = device
        self._torch = torch
        self._model = muq.MuQMuLan.from_pretrained(checkpoint)
        # The model card recommends fp32: reduced precision can produce NaNs.
        self._model.to(device=device, dtype=torch.float32)
        self._model.eval()
        self._cached_texts: tuple[str, ...] | None = None
        self._cached_text_embeddings = None

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
            else np.asarray(window, dtype=np.float32)
            for window in audio_windows
        ]
        with self._torch.inference_mode():
            text_key = tuple(texts)
            if text_key != self._cached_texts:
                self._cached_text_embeddings = self._model(texts=list(texts))
                self._cached_texts = text_key
            # The final window can be shorter than the others. Calling the official
            # model once per window preserves its own crop/tail behavior without
            # introducing zero padding or a different wraparound policy.
            audio_embeddings = self._torch.cat(
                [
                    self._model(
                        wavs=self._torch.from_numpy(window[None, :]).to(
                            self.device_name, dtype=self._torch.float32
                        )
                    )
                    for window in resampled
                ],
                dim=0,
            )
            similarities = self._model.calc_similarity(
                audio_embeddings, self._cached_text_embeddings
            )
        return np.asarray(similarities.float().cpu().numpy(), dtype=np.float64)


def _create_backend() -> AlignmentBackend:
    return MuQMuLanBackend()


class MuQMuLanAlignmentMetric:
    """Measure music-specific prompt alignment with MuQ-MuLan."""

    name = "muq_mulan_alignment"

    def __init__(
        self,
        *,
        backend_factory: BackendFactory = _create_backend,
        window_seconds: float = WINDOW_SECONDS,
        batch_windows: int = DEFAULT_BATCH_WINDOWS,
    ) -> None:
        if not math.isfinite(window_seconds) or window_seconds <= 0:
            raise ValueError("MuQ-MuLan window_seconds must be positive and finite")
        if batch_windows <= 0:
            raise ValueError("MuQ-MuLan batch_windows must be positive")
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
            label="MuQ-MuLan",
            entry=entry,
            candidate=candidate,
            backend=backend,
            window_seconds=self._window_seconds,
            batch_windows=self._batch_windows,
            metadata={
                "checkpoint": CHECKPOINT,
                "muq_version": _package_version("muq"),
                "device": getattr(backend, "device_name", "unknown"),
                "weights_license": "CC-BY-NC-4.0",
                "precision": "float32",
                "preprocessing": {
                    "model_sample_rate_hz": MODEL_SAMPLE_RATE,
                    "channels": "mono",
                    "resampler": "soxr HQ",
                    "window_seconds": self._window_seconds,
                    "hop_seconds": self._window_seconds,
                    "short_window_handling": "official MuQ crop/tail implementation",
                    "inference_batch_windows": self._batch_windows,
                    "aggregation": "duration-weighted cosine similarity",
                },
            },
        )
