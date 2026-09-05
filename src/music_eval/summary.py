from __future__ import annotations

from collections import Counter, defaultdict
from statistics import fmean
from typing import Any

from music_eval.models import EvaluationResult


def status_summary(results: list[EvaluationResult]) -> dict[str, int | float]:
    counts = Counter(result.status for result in results)
    total = len(results)
    return {
        "total": total,
        "passed": counts["pass"],
        "warnings": counts["warning"],
        "failed": counts["fail"],
        "pass_rate": counts["pass"] / total if total else 0.0,
    }


def grouped_summary(
    results: list[EvaluationResult],
) -> dict[str, dict[str, dict[str, Any]]]:
    buckets: dict[str, dict[str, list[EvaluationResult]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for result in results:
        for dimension, label_values in result.labels.items():
            for value in label_values:
                buckets[dimension][value].append(result)

    groups: dict[str, dict[str, dict[str, Any]]] = {}
    for dimension, group_values in sorted(buckets.items()):
        groups[dimension] = {}
        for value, members in sorted(group_values.items()):
            group: dict[str, Any] = status_summary(members)
            group["finding_counts"] = dict(
                sorted(
                    Counter(
                        f"{finding.source}:{finding.code}"
                        for result in members
                        for finding in result.findings
                    ).items()
                )
            )
            for metric_name in (
                "duration_seconds",
                "rms_dbfs",
                "silent_window_ratio",
                "longest_dropout_seconds",
            ):
                values_for_metric = []
                for result in members:
                    integrity = result.metrics.get("integrity")
                    if not isinstance(integrity, dict):
                        continue
                    metric_value = integrity.get(metric_name)
                    if isinstance(metric_value, (int, float)) and not isinstance(
                        metric_value, bool
                    ):
                        values_for_metric.append(float(metric_value))
                if values_for_metric:
                    group[f"mean_{metric_name}"] = fmean(values_for_metric)
            groups[dimension][value] = group
    return groups


def report_payload(results: list[EvaluationResult]) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "summary": status_summary(results),
        "groups": grouped_summary(results),
        "results": [result.to_dict() for result in results],
    }
