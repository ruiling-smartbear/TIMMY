from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

Severity = Literal["warning", "failure"]
Status = Literal["pass", "warning", "fail"]


@dataclass(frozen=True)
class Expectations:
    duration_seconds: float | None = None
    duration_tolerance_seconds: float = 0.25
    sample_rate: int | None = None
    channels: int | None = None
    max_clipped_ratio: float | None = None
    max_silent_window_ratio: float | None = None
    max_dropout_seconds: float | None = None
    max_reference_duration_delta_seconds: float | None = None
    min_reference_waveform_correlation: float | None = None
    max_reference_nrmse: float | None = None


@dataclass(frozen=True)
class ManifestEntry:
    id: str
    audio: Path
    prompt: str | None = None
    negative_prompts: tuple[str, ...] = ()
    seed: int | None = None
    reference: Path | None = None
    labels: dict[str, tuple[str, ...]] = field(default_factory=dict)
    expectations: Expectations = field(default_factory=Expectations)


@dataclass(frozen=True)
class Finding:
    code: str
    severity: Severity
    message: str
    source: str = "core"


@dataclass(frozen=True)
class Dropout:
    start_seconds: float
    end_seconds: float
    duration_seconds: float


@dataclass
class EvaluationResult:
    id: str
    audio: str
    status: Status
    metrics: dict[str, Any] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    dropouts: list[Dropout] = field(default_factory=list)
    prompt: str | None = None
    negative_prompts: tuple[str, ...] = ()
    seed: int | None = None
    reference: str | None = None
    labels: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
