from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

Direction = Literal["increase", "decrease"]


@dataclass(frozen=True)
class OrderingObservation:
    id: str
    condition: str
    metric: str
    level: float
    score: float
    expected_direction: Direction


def _required_text(item: dict[str, Any], name: str, line: int) -> str:
    value = item.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"line {line}: {name} must be a non-empty string")
    return value.strip()


def _required_number(item: dict[str, Any], name: str, line: int) -> float:
    value = item.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"line {line}: {name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"line {line}: {name} must be a finite number")
    return result


def load_ordering_manifest(path: Path) -> list[OrderingObservation]:
    observations: list[OrderingObservation] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ValueError(f"line {line}: invalid JSON: {error.msg}") from error
            if not isinstance(item, dict):
                raise ValueError(f"line {line}: observation must be an object")
            unknown = set(item) - {
                "id",
                "condition",
                "metric",
                "level",
                "score",
                "expected_direction",
            }
            if unknown:
                raise ValueError(f"line {line}: unknown fields: {', '.join(sorted(unknown))}")
            observation_id = _required_text(item, "id", line)
            if observation_id in seen:
                raise ValueError(f"line {line}: duplicate id {observation_id!r}")
            seen.add(observation_id)
            raw_direction = _required_text(item, "expected_direction", line)
            if raw_direction not in {"increase", "decrease"}:
                raise ValueError(
                    f"line {line}: expected_direction must be 'increase' or 'decrease'"
                )
            direction = cast(Direction, raw_direction)
            observations.append(
                OrderingObservation(
                    id=observation_id,
                    condition=_required_text(item, "condition", line),
                    metric=_required_text(item, "metric", line),
                    level=_required_number(item, "level", line),
                    score=_required_number(item, "score", line),
                    expected_direction=direction,
                )
            )
    if not observations:
        raise ValueError("ordering manifest is empty")
    return observations


def _kendall_tau_b(left: np.ndarray, right: np.ndarray) -> float | None:
    concordant = 0
    discordant = 0
    left_ties = 0
    right_ties = 0
    for i in range(len(left)):
        for j in range(i + 1, len(left)):
            left_delta = np.sign(left[j] - left[i])
            right_delta = np.sign(right[j] - right[i])
            if left_delta == 0 and right_delta == 0:
                continue
            if left_delta == 0:
                left_ties += 1
            elif right_delta == 0:
                right_ties += 1
            elif left_delta == right_delta:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + left_ties)
        * (concordant + discordant + right_ties)
    )
    if denominator == 0:
        return None
    return (concordant - discordant) / denominator


def analyze_metric_ordering(
    observations: list[OrderingObservation],
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[OrderingObservation]] = defaultdict(list)
    for observation in observations:
        grouped[(observation.condition, observation.metric)].append(observation)

    results: list[dict[str, Any]] = []
    for (condition, metric), group in sorted(grouped.items()):
        directions = {observation.expected_direction for observation in group}
        if len(directions) != 1:
            raise ValueError(
                f"condition {condition!r}, metric {metric!r} mixes expected directions"
            )
        direction = directions.pop()
        by_level: dict[float, list[float]] = defaultdict(list)
        for observation in group:
            by_level[observation.level].append(observation.score)
        if len(by_level) < 2:
            raise ValueError(
                f"condition {condition!r}, metric {metric!r} needs at least two levels"
            )
        levels = []
        for level, scores in sorted(by_level.items()):
            values = np.asarray(scores, dtype=np.float64)
            levels.append(
                {
                    "level": level,
                    "samples": len(scores),
                    "score_mean": float(values.mean()),
                    "score_std": float(values.std()),
                    "score_min": float(values.min()),
                    "score_max": float(values.max()),
                }
            )
        level_values = np.asarray([entry["level"] for entry in levels], dtype=np.float64)
        score_values = np.asarray([entry["score_mean"] for entry in levels], dtype=np.float64)
        adjusted_scores = score_values if direction == "increase" else -score_values
        concordant = 0
        discordant = 0
        tied = 0
        for i in range(len(adjusted_scores)):
            for j in range(i + 1, len(adjusted_scores)):
                difference = adjusted_scores[j] - adjusted_scores[i]
                if difference > 0:
                    concordant += 1
                elif difference < 0:
                    discordant += 1
                else:
                    tied += 1
        decided = concordant + discordant
        results.append(
            {
                "condition": condition,
                "metric": metric,
                "expected_direction": direction,
                "observations": len(group),
                "levels": levels,
                "kendall_tau_b": _kendall_tau_b(level_values, adjusted_scores),
                "ordered_pair_accuracy": concordant / decided if decided else None,
                "ordered_pairs": {
                    "concordant": concordant,
                    "discordant": discordant,
                    "tied": tied,
                },
            }
        )

    return {
        "schema_version": 1,
        "groups": results,
        "notes": {
            "ordering": (
                "Levels must encode increasing degradation strength. Scores are sign-adjusted "
                "from expected_direction before Kendall tau-b is calculated."
            ),
            "replicates": (
                "Replicates are averaged within each level before ordering statistics; per-level "
                "sample count, population standard deviation, minimum, and maximum are retained."
            ),
            "interpretation": (
                "High ordering agreement proves sensitivity to this perturbation family, not "
                "general perceptual validity or causal attribution."
            ),
        },
    }
