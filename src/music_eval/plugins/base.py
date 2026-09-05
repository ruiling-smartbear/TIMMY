from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from music_eval.audio import AudioData
from music_eval.models import Dropout, Finding, ManifestEntry


@dataclass(frozen=True)
class AnalysisConfig:
    silence_dbfs: float = -50.0
    window_seconds: float = 0.25
    minimum_dropout_seconds: float = 0.5
    clip_threshold: float = 0.999


@dataclass
class MetricOutput:
    metrics: dict[str, Any] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    dropouts: list[Dropout] = field(default_factory=list)


class MetricPlugin(Protocol):
    name: str

    def evaluate(
        self,
        entry: ManifestEntry,
        candidate: AudioData,
        reference: AudioData | None,
        config: AnalysisConfig,
    ) -> MetricOutput: ...
