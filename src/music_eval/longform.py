from __future__ import annotations

import json
import math
from pathlib import Path

from music_eval.suites import SUITES, suite_names

DEFAULT_LONGFORM_DURATIONS = (30.0, 120.0, 240.0, 480.0)


def write_longform_manifest(
    suite_name: str,
    output: Path,
    *,
    durations_seconds: tuple[float, ...] = DEFAULT_LONGFORM_DURATIONS,
    seeds: tuple[int, ...] = (9, 17),
    audio_directory: str = "outputs",
    case_ids: tuple[str, ...] | None = None,
    sample_rate: int | None = None,
    channels: int | None = None,
    overwrite: bool = False,
) -> int:
    """Write matched prompt/seed cases that differ only in requested duration."""
    if suite_name not in SUITES:
        raise ValueError(f"unknown suite '{suite_name}'; available: {', '.join(suite_names())}")
    if not durations_seconds:
        raise ValueError("at least one duration is required")
    if any(not math.isfinite(value) or value <= 0 for value in durations_seconds):
        raise ValueError("durations must be positive finite numbers")
    if len(set(durations_seconds)) != len(durations_seconds):
        raise ValueError("durations must be unique")
    if tuple(sorted(durations_seconds)) != durations_seconds:
        raise ValueError("durations must be strictly increasing")
    if not seeds:
        raise ValueError("at least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    if not audio_directory.strip():
        raise ValueError("audio directory must be nonempty")
    if sample_rate is not None and sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if channels is not None and channels <= 0:
        raise ValueError("channels must be positive")
    if output.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing long-form manifest: {output}")

    suite = SUITES[suite_name]
    cases_by_id = {case.id: case for case in suite.cases}
    selected_ids = tuple(cases_by_id) if case_ids is None else case_ids
    if not selected_ids:
        raise ValueError("at least one suite case is required")
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("case_ids must be unique")
    unknown = sorted(set(selected_ids) - set(cases_by_id))
    if unknown:
        raise ValueError(f"unknown case id(s): {', '.join(unknown)}")

    entries: list[dict[str, object]] = []
    for case_id in selected_ids:
        case = cases_by_id[case_id]
        for seed in seeds:
            group_id = f"{case.id}-seed{seed}"
            for duration in durations_seconds:
                duration_tag = _duration_tag(duration)
                expectations: dict[str, object] = {
                    "duration_seconds": duration,
                    "duration_tolerance_seconds": 0.25,
                }
                if sample_rate is not None:
                    expectations["sample_rate"] = sample_rate
                if channels is not None:
                    expectations["channels"] = channels
                entries.append(
                    {
                        "id": f"{group_id}-{duration_tag}",
                        "audio": (
                            f"{audio_directory.rstrip('/')}/{group_id}-{duration_tag}.wav"
                        ),
                        "prompt": case.prompt,
                        "negative_prompts": [
                            f"This is {genre} music" for genre in case.negative_genres
                        ],
                        "seed": seed,
                        "labels": {
                            "genre": case.genre,
                            "mood": list(case.mood),
                            "instruments": list(case.instruments),
                            "vocals": case.vocals,
                            "protocol": "matched-longform-v1",
                            "longform_group": group_id,
                            "target_duration_seconds": _number_label(duration),
                        },
                        "expectations": expectations,
                    }
                )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries),
        encoding="utf-8",
    )
    return len(entries)


def _duration_tag(duration: float) -> str:
    return f"d{_number_label(duration).replace('.', 'p')}s"


def _number_label(value: float) -> str:
    return str(int(value)) if value.is_integer() else format(value, ".12g")
