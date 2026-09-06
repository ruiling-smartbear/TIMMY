from __future__ import annotations

import json

import pytest

from music_eval.meta_evaluation import (
    OrderingObservation,
    analyze_metric_ordering,
    load_ordering_manifest,
)


def _observation(
    observation_id: str,
    level: float,
    score: float,
    *,
    direction: str = "increase",
) -> OrderingObservation:
    assert direction in {"increase", "decrease"}
    return OrderingObservation(
        id=observation_id,
        condition="white-noise",
        metric="distance",
        level=level,
        score=score,
        expected_direction=direction,  # type: ignore[arg-type]
    )


def test_perfect_increasing_order_has_unit_tau_and_pair_accuracy():
    result = analyze_metric_ordering(
        [_observation("a", 0.0, 0.1), _observation("b", 0.5, 0.4), _observation("c", 1.0, 0.9)]
    )["groups"][0]

    assert result["kendall_tau_b"] == pytest.approx(1.0)
    assert result["ordered_pair_accuracy"] == pytest.approx(1.0)
    assert result["ordered_pairs"] == {"concordant": 3, "discordant": 0, "tied": 0}


def test_decreasing_quality_metric_is_sign_adjusted_and_replicates_are_retained():
    result = analyze_metric_ordering(
        [
            _observation("a1", 0.0, 0.95, direction="decrease"),
            _observation("a2", 0.0, 0.85, direction="decrease"),
            _observation("b1", 1.0, 0.6, direction="decrease"),
            _observation("b2", 1.0, 0.4, direction="decrease"),
        ]
    )["groups"][0]

    assert result["kendall_tau_b"] == pytest.approx(1.0)
    assert result["levels"][0] == {
        "level": 0.0,
        "samples": 2,
        "score_mean": pytest.approx(0.9),
        "score_std": pytest.approx(0.05),
        "score_min": 0.85,
        "score_max": 0.95,
    }


def test_score_ties_are_exposed_instead_of_counted_as_correct():
    result = analyze_metric_ordering(
        [_observation("a", 0.0, 1.0), _observation("b", 1.0, 1.0)]
    )["groups"][0]

    assert result["kendall_tau_b"] is None
    assert result["ordered_pair_accuracy"] is None
    assert result["ordered_pairs"]["tied"] == 1


def test_manifest_rejects_mixed_directions_within_one_group(tmp_path):
    path = tmp_path / "ordering.jsonl"
    rows = [
        {
            "id": "one",
            "condition": "noise",
            "metric": "score",
            "level": 0,
            "score": 1,
            "expected_direction": "increase",
        },
        {
            "id": "two",
            "condition": "noise",
            "metric": "score",
            "level": 1,
            "score": 2,
            "expected_direction": "decrease",
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    with pytest.raises(ValueError, match="mixes expected directions"):
        analyze_metric_ordering(load_ordering_manifest(path))
