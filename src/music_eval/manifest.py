from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from music_eval.models import Expectations, ManifestEntry

ENTRY_FIELDS = {
    "id",
    "audio",
    "prompt",
    "negative_prompts",
    "seed",
    "reference",
    "labels",
    "expectations",
}


def _optional_path(value: Any, base: Path) -> Path | None:
    if value is None:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else base / path


def _is_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))


def _expectations(raw: Any, line_number: int, entry_id: str) -> Expectations:
    if raw is None:
        return Expectations()
    if not isinstance(raw, dict):
        raise ValueError(f"line {line_number}: expectations must be an object")
    allowed = set(Expectations.__dataclass_fields__)
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(
            f"line {line_number}: unknown expectation field(s): {', '.join(unknown)}"
        )
    # Every other expectation defaults to None (no gate); this one is a number the
    # integrity gate compares against, so null would surface later as a TypeError.
    if "duration_tolerance_seconds" in raw and raw["duration_tolerance_seconds"] is None:
        raise ValueError(
            f"line {line_number}: duration_tolerance_seconds must not be null "
            f"(entry '{entry_id}'); omit it to keep the default"
        )
    expected = Expectations(**raw)
    nonnegative = {
        "duration_seconds": expected.duration_seconds,
        "duration_tolerance_seconds": expected.duration_tolerance_seconds,
        "max_dropout_seconds": expected.max_dropout_seconds,
        "max_reference_duration_delta_seconds": (
            expected.max_reference_duration_delta_seconds
        ),
        "max_reference_nrmse": expected.max_reference_nrmse,
    }
    for name, value in nonnegative.items():
        if value is not None and (
            not _is_number(value) or not math.isfinite(value) or value < 0
        ):
            raise ValueError(f"line {line_number}: {name} must be a nonnegative number")
    for name, value in {
        "max_clipped_ratio": expected.max_clipped_ratio,
        "max_silent_window_ratio": expected.max_silent_window_ratio,
    }.items():
        if value is not None and (
            not _is_number(value) or not math.isfinite(value) or not 0.0 <= value <= 1.0
        ):
            raise ValueError(f"line {line_number}: {name} must be between 0 and 1")
    correlation = expected.min_reference_waveform_correlation
    if correlation is not None and (
        not _is_number(correlation)
        or not math.isfinite(correlation)
        or not -1.0 <= correlation <= 1.0
    ):
        raise ValueError(
            f"line {line_number}: min_reference_waveform_correlation must be between -1 and 1"
        )
    if expected.sample_rate is not None and (
        isinstance(expected.sample_rate, bool)
        or not isinstance(expected.sample_rate, int)
        or expected.sample_rate <= 0
    ):
        raise ValueError(f"line {line_number}: sample_rate must be a positive integer")
    if expected.channels is not None and (
        isinstance(expected.channels, bool)
        or not isinstance(expected.channels, int)
        or expected.channels <= 0
    ):
        raise ValueError(f"line {line_number}: channels must be a positive integer")
    return expected


def _string_list(raw: Any, field_name: str, line_number: int) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list) or any(
        not isinstance(value, str) or not value.strip() for value in raw
    ):
        raise ValueError(f"line {line_number}: {field_name} must be a list of strings")
    return tuple(dict.fromkeys(value.strip() for value in raw))


def _labels(raw: Any, line_number: int) -> dict[str, tuple[str, ...]]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"line {line_number}: labels must be an object")
    labels: dict[str, tuple[str, ...]] = {}
    for raw_dimension, raw_values in raw.items():
        if not isinstance(raw_dimension, str) or not raw_dimension.strip():
            raise ValueError(f"line {line_number}: label dimensions must be nonempty strings")
        values = [raw_values] if isinstance(raw_values, str) else raw_values
        if not isinstance(values, list) or any(
            not isinstance(value, str) or not value.strip() for value in values
        ):
            raise ValueError(
                f"line {line_number}: label '{raw_dimension}' must be a string or string list"
            )
        dimension = raw_dimension.strip().casefold()
        if dimension in labels:
            raise ValueError(f"line {line_number}: duplicate label dimension: {dimension}")
        labels[dimension] = tuple(
            dict.fromkeys(value.strip().casefold() for value in values)
        )
    return labels


def load_manifest(path: Path) -> list[ManifestEntry]:
    entries: list[ManifestEntry] = []
    ids: set[str] = set()
    base = path.resolve().parent
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"line {line_number}: invalid JSON: {error.msg}") from error
            if not isinstance(raw, dict):
                raise ValueError(f"line {line_number}: entry must be an object")
            unknown = sorted(set(raw) - ENTRY_FIELDS)
            if unknown:
                raise ValueError(
                    f"line {line_number}: unknown entry field(s): {', '.join(unknown)}"
                )
            if not isinstance(raw.get("id"), str) or not raw["id"].strip():
                raise ValueError(f"line {line_number}: id must be a nonempty string")
            if not isinstance(raw.get("audio"), str) or not raw["audio"].strip():
                raise ValueError(f"line {line_number}: audio must be a nonempty path string")
            if raw.get("reference") is not None and (
                not isinstance(raw["reference"], str) or not raw["reference"].strip()
            ):
                raise ValueError(f"line {line_number}: reference must be a path string or null")
            if raw.get("prompt") is not None and not isinstance(raw["prompt"], str):
                raise ValueError(f"line {line_number}: prompt must be a string or null")
            if raw.get("seed") is not None and (
                isinstance(raw["seed"], bool) or not isinstance(raw["seed"], int)
            ):
                raise ValueError(f"line {line_number}: seed must be an integer or null")
            entry_id = raw["id"].strip()
            if entry_id in ids:
                raise ValueError(f"line {line_number}: duplicate id: {entry_id}")
            ids.add(entry_id)
            audio = _optional_path(raw["audio"], base)
            assert audio is not None
            entries.append(
                ManifestEntry(
                    id=entry_id,
                    audio=audio,
                    prompt=raw.get("prompt"),
                    negative_prompts=_string_list(
                        raw.get("negative_prompts"), "negative_prompts", line_number
                    ),
                    seed=raw.get("seed"),
                    reference=_optional_path(raw.get("reference"), base),
                    labels=_labels(raw.get("labels"), line_number),
                    expectations=_expectations(
                        raw.get("expectations"), line_number, entry_id
                    ),
                )
            )
    if not entries:
        raise ValueError("manifest contains no entries")
    return entries
