from __future__ import annotations

import json

import numpy as np
import pytest

from music_eval.distribution import (
    EmbeddingRecord,
    compare_embedding_distributions,
    load_embedding_manifest,
)


def _record(record_id: str, system: str, values: list[float], genre: str = "rock"):
    return EmbeddingRecord(
        id=record_id,
        system=system,
        embedding=np.asarray(values, dtype=np.float64),
        embedding_model="test-encoder",
        checkpoint="revision-1",
        labels={"genre": (genre,)},
    )


def test_identical_embedding_sets_have_zero_frechet_distance():
    vectors = [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]]
    records = [
        *[_record(f"ref-{i}", "reference", vector) for i, vector in enumerate(vectors)],
        *[_record(f"same-{i}", "same", vector) for i, vector in enumerate(vectors)],
    ]

    result = compare_embedding_distributions(records, "reference", neighbors=2)
    metrics = result["comparisons"]["same"]

    assert metrics["frechet_embedding_distance"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["candidate_to_reference_nearest"]["mean"] == pytest.approx(0.0)
    assert metrics["reference_to_candidate_nearest"]["mean"] == pytest.approx(0.0)
    assert metrics["manifold"]["precision"] == 1.0
    assert metrics["manifold"]["recall"] == 1.0
    assert result["strata"]["genre"]["rock"]["same"][
        "frechet_embedding_distance"
    ] == pytest.approx(0.0, abs=1e-12)


def test_collapsed_candidate_has_lower_diversity_and_worse_coverage():
    records = [
        _record("r1", "reference", [1.0, 0.0]),
        _record("r2", "reference", [0.0, 1.0]),
        _record("r3", "reference", [-1.0, 0.0]),
        _record("r4", "reference", [0.0, -1.0]),
        _record("c1", "collapsed", [1.0, 0.0]),
        _record("c2", "collapsed", [1.0, 0.0]),
        _record("c3", "collapsed", [1.0, 0.0]),
        _record("c4", "collapsed", [1.0, 0.0]),
    ]

    metrics = compare_embedding_distributions(records, "reference", neighbors=1)[
        "comparisons"
    ]["collapsed"]

    assert metrics["candidate_diversity_mean_cosine_distance"] == 0.0
    assert metrics["reference_diversity_mean_cosine_distance"] > 0.0
    assert metrics["reference_to_candidate_nearest"]["mean"] > 0.0
    assert metrics["frechet_embedding_distance"] > 0.0


def test_embedding_manifest_rejects_mixed_checkpoint_provenance(tmp_path):
    path = tmp_path / "embeddings.jsonl"
    common = {"embedding_model": "encoder", "labels": {"genre": "rock"}}
    path.write_text(
        json.dumps(
            {"id": "a", "system": "one", "embedding": [1, 0], "checkpoint": "v1", **common}
        )
        + "\n"
        + json.dumps(
            {"id": "b", "system": "two", "embedding": [0, 1], "checkpoint": "v2", **common}
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="provenance"):
        load_embedding_manifest(path)


def test_manifold_metrics_are_omitted_for_too_few_samples():
    records = [
        _record("r1", "reference", [1.0, 0.0]),
        _record("r2", "reference", [0.0, 1.0]),
        _record("c1", "candidate", [0.9, 0.1]),
        _record("c2", "candidate", [0.1, 0.9]),
    ]

    metrics = compare_embedding_distributions(records, "reference", neighbors=3)[
        "comparisons"
    ]["candidate"]

    assert metrics["manifold"] is None
