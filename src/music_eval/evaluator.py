from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

from music_eval.audio import read_wav
from music_eval.models import EvaluationResult, Finding, ManifestEntry, Status
from music_eval.plugins.base import AnalysisConfig, MetricPlugin
from music_eval.plugins.registry import resolve_plugins


def _status(findings: list[Finding]) -> Status:
    if any(finding.severity == "failure" for finding in findings):
        return "fail"
    if findings:
        return "warning"
    return "pass"


def _failed_result(entry: ManifestEntry, code: str, message: str) -> EvaluationResult:
    return EvaluationResult(
        id=entry.id,
        audio=str(entry.audio),
        reference=str(entry.reference) if entry.reference else None,
        status="fail",
        findings=[Finding(code, "failure", message)],
        prompt=entry.prompt,
        negative_prompts=entry.negative_prompts,
        seed=entry.seed,
        labels=entry.labels,
    )


def evaluate_entry(
    entry: ManifestEntry,
    *,
    config: AnalysisConfig | None = None,
    plugins: Iterable[MetricPlugin] | None = None,
    silence_dbfs: float = -50.0,
    window_seconds: float = 0.25,
    minimum_dropout_seconds: float = 0.5,
    clip_threshold: float = 0.999,
) -> EvaluationResult:
    """Evaluate one candidate.

    Individual threshold arguments preserve the v0.1 API. New callers should
    pass an ``AnalysisConfig``.
    """
    active_config = config or AnalysisConfig(
        silence_dbfs=silence_dbfs,
        window_seconds=window_seconds,
        minimum_dropout_seconds=minimum_dropout_seconds,
        clip_threshold=clip_threshold,
    )
    try:
        candidate = read_wav(entry.audio)
    except (OSError, ValueError) as error:
        return _failed_result(entry, "wav_decode", str(error))

    reference = None
    if entry.reference is not None:
        try:
            reference = read_wav(entry.reference)
        except (OSError, ValueError) as error:
            return _failed_result(entry, "reference_wav_decode", str(error))

    active_plugins = list(plugins or resolve_plugins())
    plugin_names = [getattr(plugin, "name", None) for plugin in active_plugins]
    if any(not isinstance(name, str) or not name for name in plugin_names):
        raise ValueError("every metric plugin must have a nonempty string name")
    if len(set(plugin_names)) != len(plugin_names):
        raise ValueError("metric plugin names must be unique")

    metrics: dict[str, object] = {}
    findings: list[Finding] = []
    dropouts = []
    for plugin in active_plugins:
        try:
            output = plugin.evaluate(entry, candidate, reference, active_config)
        except Exception as error:  # A plugin failure is data in a batch evaluation.
            findings.append(
                Finding(
                    "plugin_error",
                    "failure",
                    f"{type(error).__name__}: {error}",
                    source=plugin.name,
                )
            )
            continue
        metrics[plugin.name] = output.metrics
        findings.extend(replace(finding, source=plugin.name) for finding in output.findings)
        dropouts.extend(output.dropouts)

    return EvaluationResult(
        id=entry.id,
        audio=str(entry.audio),
        reference=str(entry.reference) if entry.reference else None,
        status=_status(findings),
        metrics=metrics,
        findings=findings,
        dropouts=dropouts,
        prompt=entry.prompt,
        negative_prompts=entry.negative_prompts,
        seed=entry.seed,
        labels=entry.labels,
    )


def evaluate_manifest(
    entries: list[ManifestEntry],
    *,
    config: AnalysisConfig | None = None,
    plugins: Iterable[MetricPlugin] | None = None,
) -> list[EvaluationResult]:
    active_plugins = list(plugins or resolve_plugins())
    return [
        evaluate_entry(entry, config=config, plugins=active_plugins) for entry in entries
    ]


def ensure_output_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
