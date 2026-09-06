from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class EmbeddingRecord:
    id: str
    system: str
    embedding: NDArray[np.float64]
    embedding_model: str
    checkpoint: str
    labels: dict[str, tuple[str, ...]]


def _text(item: dict[str, Any], name: str, line: int) -> str:
    value = item.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"line {line}: {name} must be a non-empty string")
    return value.strip()


def _labels(value: object, line: int) -> dict[str, tuple[str, ...]]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"line {line}: labels must be an object")
    output: dict[str, tuple[str, ...]] = {}
    for raw_key, raw_values in value.items():
        if not isinstance(raw_key, str) or not raw_key.strip():
            raise ValueError(f"line {line}: label names must be non-empty strings")
        values = [raw_values] if isinstance(raw_values, str) else raw_values
        if not isinstance(values, list) or not values or not all(
            isinstance(item, str) and item.strip() for item in values
        ):
            raise ValueError(f"line {line}: label values must be strings or string lists")
        output[raw_key.strip().casefold()] = tuple(
            dict.fromkeys(item.strip().casefold() for item in values)
        )
    return output


def load_embedding_manifest(path: Path) -> list[EmbeddingRecord]:
    records: list[EmbeddingRecord] = []
    seen: set[str] = set()
    expected_provenance: tuple[str, str] | None = None
    expected_dimension: int | None = None
    with path.open(encoding="utf-8") as handle:
        for line, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ValueError(f"line {line}: invalid JSON: {error.msg}") from error
            if not isinstance(item, dict):
                raise ValueError(f"line {line}: embedding record must be an object")
            unknown = set(item) - {
                "id",
                "system",
                "embedding",
                "embedding_model",
                "checkpoint",
                "labels",
            }
            if unknown:
                raise ValueError(f"line {line}: unknown fields: {', '.join(sorted(unknown))}")
            record_id = _text(item, "id", line)
            if record_id in seen:
                raise ValueError(f"line {line}: duplicate id {record_id!r}")
            seen.add(record_id)
            raw_embedding = item.get("embedding")
            if not isinstance(raw_embedding, list) or not raw_embedding:
                raise ValueError(f"line {line}: embedding must be a non-empty number list")
            if not all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in raw_embedding
            ):
                raise ValueError(f"line {line}: embedding must contain only numbers")
            embedding = np.asarray(raw_embedding, dtype=np.float64)
            if not np.isfinite(embedding).all() or np.linalg.norm(embedding) == 0:
                raise ValueError(f"line {line}: embedding must be finite and non-zero")
            provenance = (_text(item, "embedding_model", line), _text(item, "checkpoint", line))
            if expected_provenance is None:
                expected_provenance = provenance
                expected_dimension = embedding.size
            if provenance != expected_provenance:
                raise ValueError(
                    f"line {line}: embedding provenance does not match earlier records"
                )
            if embedding.size != expected_dimension:
                raise ValueError(f"line {line}: embedding dimension does not match earlier records")
            records.append(
                EmbeddingRecord(
                    id=record_id,
                    system=_text(item, "system", line),
                    embedding=embedding,
                    embedding_model=provenance[0],
                    checkpoint=provenance[1],
                    labels=_labels(item.get("labels"), line),
                )
            )
    if not records:
        raise ValueError("embedding manifest is empty")
    return records


def _cosine_distances(left: NDArray[np.float64], right: NDArray[np.float64]) -> NDArray[np.float64]:
    left_normalized = left / np.linalg.norm(left, axis=1, keepdims=True)
    right_normalized = right / np.linalg.norm(right, axis=1, keepdims=True)
    return np.asarray(
        np.clip(1.0 - left_normalized @ right_normalized.T, 0.0, 2.0),
        dtype=np.float64,
    )


def _frechet_distance(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
    left_mean, right_mean = left.mean(axis=0), right.mean(axis=0)
    left_cov = np.atleast_2d(np.cov(left, rowvar=False))
    right_cov = np.atleast_2d(np.cov(right, rowvar=False))
    eigenvalues, eigenvectors = np.linalg.eigh(left_cov)
    left_sqrt = (eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))) @ eigenvectors.T
    middle = left_sqrt @ right_cov @ left_sqrt
    trace_sqrt = float(np.sqrt(np.clip(np.linalg.eigvalsh(middle), 0.0, None)).sum())
    distance = (
        float(np.dot(left_mean - right_mean, left_mean - right_mean))
        + float(np.trace(left_cov))
        + float(np.trace(right_cov))
        - 2.0 * trace_sqrt
    )
    return 0.0 if abs(distance) <= 1e-12 else max(0.0, distance)


def _mean_pairwise_distance(values: NDArray[np.float64]) -> float:
    distances = _cosine_distances(values, values)
    upper = distances[np.triu_indices(len(values), k=1)]
    return float(upper.mean()) if upper.size else 0.0


def _manifold_metrics(
    candidate: NDArray[np.float64], reference: NDArray[np.float64], neighbors: int
) -> dict[str, float] | None:
    if len(candidate) <= neighbors or len(reference) <= neighbors:
        return None
    within_candidate = _cosine_distances(candidate, candidate)
    within_reference = _cosine_distances(reference, reference)
    candidate_radii = np.partition(within_candidate, neighbors, axis=1)[:, neighbors]
    reference_radii = np.partition(within_reference, neighbors, axis=1)[:, neighbors]
    cross = _cosine_distances(candidate, reference)
    precision_hits = cross <= reference_radii[None, :]
    recall_hits = cross <= candidate_radii[:, None]
    return {
        "neighbors": neighbors,
        "precision": float(precision_hits.any(axis=1).mean()),
        "recall": float(recall_hits.any(axis=0).mean()),
        "density": float((precision_hits.sum(axis=1) / neighbors).mean()),
        "coverage": float((cross.min(axis=0) <= reference_radii).mean()),
    }


def _compare_arrays(
    candidate: NDArray[np.float64], reference: NDArray[np.float64], neighbors: int
) -> dict[str, Any]:
    if len(candidate) < 2 or len(reference) < 2:
        raise ValueError("distribution comparison requires at least two samples per side")
    cross = _cosine_distances(candidate, reference)
    candidate_nearest = cross.min(axis=1)
    reference_nearest = cross.min(axis=0)
    return {
        "candidate_samples": len(candidate),
        "reference_samples": len(reference),
        "frechet_embedding_distance": _frechet_distance(candidate, reference),
        "candidate_diversity_mean_cosine_distance": _mean_pairwise_distance(candidate),
        "reference_diversity_mean_cosine_distance": _mean_pairwise_distance(reference),
        "candidate_to_reference_nearest": {
            "mean": float(candidate_nearest.mean()),
            "p95": float(np.percentile(candidate_nearest, 95)),
        },
        "reference_to_candidate_nearest": {
            "mean": float(reference_nearest.mean()),
            "p95": float(np.percentile(reference_nearest, 95)),
        },
        "manifold": _manifold_metrics(candidate, reference, neighbors),
    }


def compare_embedding_distributions(
    records: list[EmbeddingRecord], reference_system: str, *, neighbors: int = 3
) -> dict[str, Any]:
    if neighbors <= 0:
        raise ValueError("neighbors must be positive")
    by_system: dict[str, list[EmbeddingRecord]] = defaultdict(list)
    for record in records:
        by_system[record.system].append(record)
    if reference_system not in by_system:
        raise ValueError(f"reference system not found: {reference_system}")
    candidates = sorted(set(by_system) - {reference_system})
    if not candidates:
        raise ValueError("manifest must contain at least one non-reference system")
    reference = by_system[reference_system]

    comparisons: dict[str, Any] = {}
    for system in candidates:
        comparisons[system] = _compare_arrays(
            np.stack([record.embedding for record in by_system[system]]),
            np.stack([record.embedding for record in reference]),
            neighbors,
        )

    buckets: dict[str, dict[str, dict[str, list[EmbeddingRecord]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for record in records:
        for dimension, values in record.labels.items():
            for value in values:
                buckets[dimension][value][record.system].append(record)
    strata: dict[str, Any] = {}
    for dimension, group_values in sorted(buckets.items()):
        strata[dimension] = {}
        for value, systems in sorted(group_values.items()):
            value_results = {}
            for system in candidates:
                candidate_group = systems.get(system, [])
                reference_group = systems.get(reference_system, [])
                if len(candidate_group) < 2 or len(reference_group) < 2:
                    continue
                value_results[system] = _compare_arrays(
                    np.stack([record.embedding for record in candidate_group]),
                    np.stack([record.embedding for record in reference_group]),
                    neighbors,
                )
            if value_results:
                strata[dimension][value] = value_results
        if not strata[dimension]:
            del strata[dimension]
    return {
        "schema_version": 1,
        "embedding_model": records[0].embedding_model,
        "checkpoint": records[0].checkpoint,
        "embedding_dimension": int(records[0].embedding.size),
        "reference_system": reference_system,
        "neighbors": neighbors,
        "comparisons": comparisons,
        "strata": strata,
        "notes": {
            "frechet": (
                "Gaussian distance in the declared embedding space; not FAD unless the "
                "embedding and protocol define FAD."
            ),
            "manifold": (
                "Precision, recall, density, and coverage are omitted when either side "
                "has at most k samples."
            ),
        },
    }
