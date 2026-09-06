from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
import uuid
import wave
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

Choice = Literal["A", "B", "tie"]
DEFAULT_CRITERIA = (
    "overall_preference",
    "prompt_alignment",
    "musicality_structure",
    "production_quality",
)

# Criteria sets taken from published listening protocols. The judgment here is
# always pairwise A/B/tie; SongEval rated songs in isolation on a 1-5 scale, so
# its dimensions are reused, not its scale.
CRITERIA_PRESETS: dict[str, tuple[str, ...]] = {
    "timmy": DEFAULT_CRITERIA,
    # Huang et al. 2025 (MAD / MusicPrefs): one preference per axis.
    "musicprefs": ("fidelity", "musicality"),
    # Yao et al. 2025 (SongEval): five aesthetic dimensions for full songs.
    "songeval": (
        "overall_coherence",
        "memorability",
        "vocal_naturalness",
        "structure_clarity",
        "overall_musicality",
    ),
    # SongEval without the vocal dimension, for instrumental material.
    "songeval-instrumental": (
        "overall_coherence",
        "memorability",
        "structure_clarity",
        "overall_musicality",
    ),
}

# What raters are told each criterion means; shown under the criterion name.
CRITERIA_DEFINITIONS: dict[str, str] = {
    "overall_preference": "Which one would you rather listen to again?",
    "prompt_alignment": "Which one better matches the prompt shown above?",
    "musicality_structure": "Which one has the stronger melody, harmony and form?",
    "production_quality": "Which one sounds better produced: mix, clarity, no artifacts?",
    "fidelity": "Which one has better sound quality, ignoring the music itself?",
    "musicality": "Which one is musically better, ignoring sound quality?",
    "overall_coherence": (
        "Which one keeps musical and emotional continuity across its sections?"
    ),
    "memorability": "Which one has the catchier melody, motif or hook?",
    "vocal_naturalness": "Which one has more natural vocal phrasing and breathing?",
    "structure_clarity": "In which one are verse, chorus and bridge clearer?",
    "overall_musicality": (
        "Which one is more enjoyable overall: melody, harmony, arrangement, vocals?"
    ),
}


@dataclass(frozen=True)
class Comparison:
    id: str
    prompt: str
    audio_a: Path
    system_a: str
    audio_b: Path
    system_b: str
    labels: dict[str, str]


def _require_string(value: object, field: str, line: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"line {line}: {field} must be a non-empty string")
    return value.strip()


def load_comparisons(path: Path) -> list[Comparison]:
    comparisons: list[Comparison] = []
    seen: set[str] = set()
    base = path.resolve().parent
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ValueError(f"line {line_number}: invalid JSON: {error.msg}") from error
            if not isinstance(item, dict):
                raise ValueError(f"line {line_number}: comparison must be an object")
            comparison_id = _require_string(item.get("id"), "id", line_number)
            if comparison_id in seen:
                raise ValueError(f"line {line_number}: duplicate id {comparison_id!r}")
            seen.add(comparison_id)
            labels = item.get("labels", {})
            if not isinstance(labels, dict) or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in labels.items()
            ):
                raise ValueError(f"line {line_number}: labels must map strings to strings")
            audio_a = base / _require_string(item.get("audio_a"), "audio_a", line_number)
            audio_b = base / _require_string(item.get("audio_b"), "audio_b", line_number)
            for audio in (audio_a, audio_b):
                if not audio.is_file():
                    raise ValueError(f"line {line_number}: audio file not found: {audio}")
            system_a = _require_string(item.get("system_a"), "system_a", line_number)
            system_b = _require_string(item.get("system_b"), "system_b", line_number)
            if system_a == system_b:
                raise ValueError(f"line {line_number}: systems in a comparison must differ")
            comparisons.append(
                Comparison(
                    id=comparison_id,
                    prompt=_require_string(item.get("prompt"), "prompt", line_number),
                    audio_a=audio_a.resolve(),
                    system_a=system_a,
                    audio_b=audio_b.resolve(),
                    system_b=system_b,
                    labels=dict(labels),
                )
            )
    if not comparisons:
        raise ValueError("comparison manifest is empty")
    return comparisons


def _copy_audio(path: Path, audio_dir: Path) -> str:
    if path.suffix.lower() != ".wav":
        raise ValueError(f"listening studies currently require PCM WAV audio: {path}")
    try:
        with wave.open(str(path), "rb") as source:
            params = source.getparams()
            frames = source.readframes(source.getnframes())
    except (EOFError, wave.Error) as error:
        raise ValueError(f"invalid PCM WAV audio: {path}: {error}") from error
    identity = (
        f"{params.nchannels}:{params.sampwidth}:{params.framerate}:{params.nframes}".encode()
        + frames
    )
    name = f"clip-{hashlib.sha256(identity).hexdigest()[:20]}.wav"
    target = audio_dir / name
    if not target.exists():
        with wave.open(str(target), "wb") as destination:
            destination.setparams(params)
            destination.writeframes(frames)
    return f"audio/{name}"


def _trial_id(study_id: str, comparison_id: str, occurrence: int) -> str:
    digest = hashlib.sha256(
        f"{study_id}\0{comparison_id}\0{occurrence}".encode()
    ).hexdigest()[:16]
    return f"trial-{digest}"


def build_listening_study(
    comparisons: Iterable[Comparison],
    output_dir: Path,
    *,
    title: str,
    seed: int,
    repeat_fraction: float,
    criteria: tuple[str, ...] = DEFAULT_CRITERIA,
    overwrite: bool = False,
) -> tuple[Path, Path, int]:
    items = list(comparisons)
    key_path = output_dir.parent / f"{output_dir.name}.organizer.json"
    if not 0.0 <= repeat_fraction <= 1.0:
        raise ValueError("repeat_fraction must be between 0 and 1")
    if not criteria or len(set(criteria)) != len(criteria):
        raise ValueError("criteria must be non-empty and unique")
    if any(not criterion or not criterion.replace("_", "").isalnum() for criterion in criteria):
        raise ValueError("criteria may contain only letters, numbers, and underscores")
    if output_dir.exists():
        if not overwrite:
            raise ValueError(f"output already exists: {output_dir}")
        shutil.rmtree(output_dir)
    if key_path.exists() and not overwrite:
        raise ValueError(f"organizer key already exists: {key_path}")
    output_dir.mkdir(parents=True)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir()

    source_fingerprint = json.dumps(
        [(item.id, item.system_a, item.system_b) for item in items], separators=(",", ":")
    )
    study_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"music-eval:{seed}:{source_fingerprint}"))
    rng = random.Random(seed)
    repeat_count = round(len(items) * repeat_fraction)
    if repeat_fraction > 0 and repeat_count == 0:
        repeat_count = 1
    repeated_ids = {item.id for item in rng.sample(items, repeat_count)}

    public_trials: list[dict[str, Any]] = []
    key_trials: list[dict[str, Any]] = []
    for item in items:
        occurrences = 2 if item.id in repeated_ids else 1
        for occurrence in range(occurrences):
            left_is_a = bool(rng.getrandbits(1))
            sides = (
                ((item.audio_a, item.system_a), (item.audio_b, item.system_b))
                if left_is_a
                else ((item.audio_b, item.system_b), (item.audio_a, item.system_a))
            )
            trial_id = _trial_id(study_id, item.id, occurrence)
            public_trials.append(
                {
                    "id": trial_id,
                    "prompt": item.prompt,
                    "audio_a": _copy_audio(sides[0][0], audio_dir),
                    "audio_b": _copy_audio(sides[1][0], audio_dir),
                }
            )
            key_trials.append(
                {
                    "id": trial_id,
                    "source_id": item.id,
                    "repeat_of": item.id if occurrence else None,
                    "system_a": sides[0][1],
                    "system_b": sides[1][1],
                    "labels": item.labels,
                }
            )
    order = list(range(len(public_trials)))
    rng.shuffle(order)
    public_trials = [public_trials[index] for index in order]
    key_by_id = {trial["id"]: trial for trial in key_trials}
    key_trials = [key_by_id[trial["id"]] for trial in public_trials]

    public = {
        "schema_version": 1,
        "study_id": study_id,
        "title": title,
        "criteria": list(criteria),
        "criteria_definitions": {
            name: CRITERIA_DEFINITIONS[name] for name in criteria if name in CRITERIA_DEFINITIONS
        },
        "trials": public_trials,
    }
    key = {
        "schema_version": 1,
        "study_id": study_id,
        "title": title,
        "seed": seed,
        "criteria": list(criteria),
        "trials": key_trials,
    }
    public_path = output_dir / "study.json"
    public_path.write_text(
        json.dumps(public, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    key_path.write_text(json.dumps(key, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return public_path, key_path, len(public_trials)


def _response_choice(value: object, context: str) -> Choice:
    if value not in {"A", "B", "tie"}:
        raise ValueError(f"{context}: choice must be A, B, or tie")
    return value


def validate_response(response: object, key: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(response, dict):
        raise ValueError("response must be an object")
    if response.get("study_id") != key["study_id"]:
        raise ValueError("response study_id does not match organizer key")
    rater_id = response.get("rater_id")
    if not isinstance(rater_id, str) or not rater_id.strip() or len(rater_id) > 100:
        raise ValueError("rater_id must be a non-empty string of at most 100 characters")
    answers = response.get("answers")
    if not isinstance(answers, list):
        raise ValueError("answers must be a list")
    submitted_at = response.get("submitted_at")
    if submitted_at is not None and not isinstance(submitted_at, str):
        raise ValueError("submitted_at must be a string")
    expected_trials = {trial["id"] for trial in key["trials"]}
    expected_criteria = set(key["criteria"])
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, answer in enumerate(answers):
        context = f"answer {index + 1}"
        if not isinstance(answer, dict) or not isinstance(answer.get("trial_id"), str):
            raise ValueError(f"{context}: invalid answer")
        trial_id = answer["trial_id"]
        if trial_id not in expected_trials or trial_id in seen:
            raise ValueError(f"{context}: unknown or duplicate trial_id")
        seen.add(trial_id)
        ratings = answer.get("ratings")
        if not isinstance(ratings, dict) or set(ratings) != expected_criteria:
            raise ValueError(f"{context}: ratings must contain every configured criterion")
        note = answer.get("note")
        normalized.append(
            {
                "trial_id": trial_id,
                "ratings": {
                    criterion: _response_choice(ratings[criterion], context)
                    for criterion in key["criteria"]
                },
                "note": "" if note is None else str(note)[:1000],
            }
        )
    if seen != expected_trials:
        raise ValueError(f"response is incomplete: missing {len(expected_trials - seen)} trial(s)")
    return {
        "schema_version": 1,
        "study_id": key["study_id"],
        "rater_id": rater_id.strip(),
        "submitted_at": submitted_at or datetime.now(timezone.utc).isoformat(),
        "answers": normalized,
    }


def _winner(choice: Choice, trial: dict[str, Any]) -> str | None:
    if choice == "tie":
        return None
    return str(trial[f"system_{choice.lower()}"])


def _strongly_connected(edges: NDArray[np.bool_], present: NDArray[np.bool_]) -> bool:
    if present.sum() < 2:
        return True
    start = int(np.flatnonzero(present)[0])
    for adjacency in (edges, edges.T):
        reached = np.zeros(len(edges), dtype=bool)
        reached[start] = True
        while True:
            frontier = adjacency[reached].any(axis=0) & ~reached
            if not frontier.any():
                break
            reached |= frontier
        if not np.array_equal(reached, present):
            return False
    return True


def _fit_bradley_terry(
    systems: list[str], observations: list[tuple[int, int, float]]
) -> list[dict[str, Any]]:
    count = len(systems)
    if count == 1:
        return [
            {
                "system": systems[0],
                "rank": 1,
                "log_strength": 0.0,
                "vs_average": 0.5,
                "separated": False,
            }
        ]
    # Aggregate observations once into count x count matrices indexed [left, right]:
    # how often the pair was seen and the total outcome (1 win, 0.5 tie, 0 loss) for left.
    table = np.array(observations, dtype=np.float64).reshape(-1, 3)
    codes = table[:, 0].astype(np.intp) * count + table[:, 1].astype(np.intp)
    seen = np.bincount(codes, minlength=count * count).reshape(count, count).astype(np.float64)
    won = np.bincount(codes, weights=table[:, 2], minlength=count * count).reshape(count, count)
    # The maximum-likelihood estimate exists iff the digraph with an edge i -> j whenever
    # i took any outcome > 0 against j is strongly connected among the compared systems.
    edges = (won > 0.0) | ((seen - won).T > 0.0)
    present = (seen.sum(axis=1) + seen.sum(axis=0)) > 0.0
    separated = not _strongly_connected(edges, present)
    theta = np.zeros(count, dtype=np.float64)
    ridge = 1e-6
    for _ in range(100):
        difference = np.clip(theta[:, None] - theta[None, :], -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-difference))
        residual = won - seen * probability
        weight = seen * probability * (1.0 - probability)
        gradient = -ridge * theta + residual.sum(axis=1) - residual.sum(axis=0)
        information = (
            np.diag(weight.sum(axis=1) + weight.sum(axis=0) + ridge) - weight - weight.T
        )
        delta = np.linalg.solve(information + np.ones_like(information) / count, gradient)
        theta += delta
        theta -= theta.mean()
        if float(np.max(np.abs(delta))) < 1e-9:
            break
    strengths = [round(float(value), 6) for value in theta]
    dense_rank = {value: rank for rank, value in enumerate(sorted(set(strengths), reverse=True), 1)}
    return [
        {
            "system": system,
            "rank": dense_rank[strengths[index]],
            "log_strength": strengths[index],
            "vs_average": round(float(1.0 / (1.0 + math.exp(-theta[index]))), 6),
            "separated": separated,
        }
        for index, system in enumerate(systems)
    ]


def _comparison_components(
    systems: list[str], trials: Iterable[dict[str, Any]]
) -> list[list[str]]:
    neighbors: dict[str, set[str]] = {system: set() for system in systems}
    for trial in trials:
        left, right = str(trial["system_a"]), str(trial["system_b"])
        neighbors[left].add(right)
        neighbors[right].add(left)
    components: list[list[str]] = []
    remaining = set(systems)
    while remaining:
        stack = [min(remaining)]
        component: set[str] = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(neighbors[current] - component)
        remaining -= component
        components.append(sorted(component))
    return components


def _rank_trials(
    trials: dict[str, dict[str, Any]],
    validated: list[dict[str, Any]],
    criteria: list[str],
) -> dict[str, Any]:
    systems = sorted(
        {str(trial[side]) for trial in trials.values() for side in ("system_a", "system_b")}
    )
    system_index = {system: index for index, system in enumerate(systems)}
    components = _comparison_components(systems, trials.values())
    component_for = {
        system: component_index
        for component_index, component in enumerate(components, 1)
        for system in component
    }
    rankings: dict[str, Any] = {}
    for criterion in criteria:
        observations: list[tuple[int, int, float]] = []
        wins = {system: 0.0 for system in systems}
        appearances = {system: 0 for system in systems}
        for response in validated:
            for answer in response["answers"]:
                if answer["trial_id"] not in trials:
                    continue
                trial = trials[answer["trial_id"]]
                left, right = str(trial["system_a"]), str(trial["system_b"])
                choice = answer["ratings"][criterion]
                outcome = 1.0 if choice == "A" else 0.0 if choice == "B" else 0.5
                observations.append((system_index[left], system_index[right], outcome))
                wins[left] += outcome
                wins[right] += 1.0 - outcome
                appearances[left] += 1
                appearances[right] += 1
        ranking = _fit_bradley_terry(systems, observations)
        for row in ranking:
            system = str(row["system"])
            row["preference_rate"] = round(wins[system] / appearances[system], 6)
            row["comparisons"] = appearances[system]
            row["component"] = component_for[system]
        if len(components) > 1:
            ranking = []
            for component_index, component in enumerate(components, 1):
                local_index = {system: index for index, system in enumerate(component)}
                local_observations = [
                    (local_index[systems[left]], local_index[systems[right]], outcome)
                    for left, right, outcome in observations
                    if systems[left] in local_index and systems[right] in local_index
                ]
                local_ranking = _fit_bradley_terry(component, local_observations)
                for row in local_ranking:
                    system = str(row["system"])
                    row["preference_rate"] = round(wins[system] / appearances[system], 6)
                    row["comparisons"] = appearances[system]
                    row["component"] = component_index
                    row["rank"] = None
                ranking.extend(local_ranking)
        rankings[criterion] = sorted(
            ranking,
            key=lambda row: (int(row["component"]), -(float(row["log_strength"]))),
        )
    return {
        "systems": systems,
        "comparison_graph_connected": len(components) == 1,
        "comparison_components": components,
        "rankings": rankings,
    }


def _bootstrap_rankings(
    aggregate: dict[str, Any],
    trials: dict[str, dict[str, Any]],
    validated: list[dict[str, Any]],
    criteria: list[str],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if samples < 0:
        raise ValueError("bootstrap_samples must be nonnegative")
    if samples == 0:
        return {"enabled": False, "reason": "disabled", "samples": 0, "seed": seed}
    if len(validated) < 2:
        return {
            "enabled": False,
            "reason": "at least two raters are required",
            "samples": 0,
            "seed": seed,
        }
    values: dict[str, dict[str, dict[str, list[float]]]] = {
        criterion: {
            system: {"log_strength": [], "preference_rate": []}
            for system in aggregate["systems"]
        }
        for criterion in criteria
    }
    rng = np.random.default_rng(seed)
    for _ in range(samples):
        indices = rng.integers(0, len(validated), size=len(validated))
        resampled = [validated[int(index)] for index in indices]
        estimate = _rank_trials(trials, resampled, criteria)
        for criterion, rows in estimate["rankings"].items():
            for row in rows:
                system_values = values[criterion][str(row["system"])]
                system_values["log_strength"].append(float(row["log_strength"]))
                system_values["preference_rate"].append(float(row["preference_rate"]))
    for criterion, rows in aggregate["rankings"].items():
        for row in rows:
            system_values = values[criterion][str(row["system"])]
            for metric in ("log_strength", "preference_rate"):
                low, high = np.percentile(system_values[metric], [2.5, 97.5])
                row[f"{metric}_ci95"] = [round(float(low), 6), round(float(high), 6)]
    return {
        "enabled": True,
        "unit": "rater",
        "samples": samples,
        "seed": seed,
        "interval": "percentile_95",
        "captures": "rater sampling uncertainty only",
    }


def _inter_rater_agreement(
    trials: dict[str, dict[str, Any]],
    validated: list[dict[str, Any]],
    criteria: list[str],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    answer_maps = [
        {answer["trial_id"]: answer for answer in response["answers"]}
        for response in validated
    ]
    for criterion in criteria:
        matching_pairs = 0
        rater_pairs = 0
        unanimous_trials = 0
        for trial_id, trial in trials.items():
            choices = []
            for answer_map in answer_maps:
                answer = answer_map[trial_id]
                winner = _winner(answer["ratings"][criterion], trial)
                choices.append(winner if winner is not None else "__tie__")
            unanimous_trials += len(set(choices)) == 1
            for left in range(len(choices)):
                for right in range(left + 1, len(choices)):
                    matching_pairs += choices[left] == choices[right]
                    rater_pairs += 1
        output[criterion] = {
            "matching_pairs": matching_pairs,
            "rater_pairs": rater_pairs,
            "pairwise_agreement_rate": (
                round(matching_pairs / rater_pairs, 6) if rater_pairs else None
            ),
            "unanimous_trials": unanimous_trials if rater_pairs else None,
            "trials": len(trials),
            "unanimous_trial_rate": (
                round(unanimous_trials / len(trials), 6) if rater_pairs else None
            ),
        }
    return output


def _side_choice_diagnostics(
    validated: list[dict[str, Any]], criteria: list[str]
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for criterion in criteria:
        counts = {"A": 0, "B": 0, "tie": 0}
        for response in validated:
            for answer in response["answers"]:
                counts[answer["ratings"][criterion]] += 1
        non_ties = counts["A"] + counts["B"]
        output[criterion] = {
            "counts": counts,
            "side_a_rate_among_non_ties": (
                round(counts["A"] / non_ties, 6) if non_ties else None
            ),
            "absolute_side_imbalance": (
                round(abs(counts["A"] - counts["B"]) / non_ties, 6)
                if non_ties
                else None
            ),
        }
    return output


def _validate_organizer_key(key: object) -> dict[str, Any]:
    if (
        not isinstance(key, dict)
        or not isinstance(key.get("study_id"), str)
        or not isinstance(key.get("title"), str)
        or not isinstance(key.get("criteria"), list)
        or not all(isinstance(criterion, str) for criterion in key["criteria"])
        or not isinstance(key.get("trials"), list)
        or not key["trials"]
    ):
        raise ValueError(
            "organizer key must be an object with study_id, title, criteria, and trials"
        )
    fields = ("id", "source_id", "system_a", "system_b")
    for index, trial in enumerate(key["trials"], 1):
        if not isinstance(trial, dict) or not all(isinstance(trial.get(f), str) for f in fields):
            hint = (
                "; this looks like the public study file (study.json), "
                "pass the private .organizer.json key instead"
                if isinstance(trial, dict) and "audio_a" in trial
                else ""
            )
            raise ValueError(f"organizer key trial {index} must have {', '.join(fields)}{hint}")
    return key


def analyze_listening_responses(
    key: dict[str, Any],
    responses: Iterable[dict[str, Any]],
    *,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 20260906,
) -> dict[str, Any]:
    key = _validate_organizer_key(key)
    trials = {trial["id"]: trial for trial in key["trials"]}
    validated = [validate_response(response, key) for response in responses]
    if not validated:
        raise ValueError("no responses supplied")
    rater_ids = [str(response["rater_id"]) for response in validated]
    if len(set(rater_ids)) != len(rater_ids):
        raise ValueError("duplicate rater_id across response files")
    aggregate = _rank_trials(trials, validated, key["criteria"])
    bootstrap = _bootstrap_rankings(
        aggregate,
        trials,
        validated,
        key["criteria"],
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )

    repeat_trials: dict[str, list[str]] = defaultdict(list)
    for trial in trials.values():
        repeat_trials[str(trial["source_id"])].append(str(trial["id"]))
    repeated = {source: ids for source, ids in repeat_trials.items() if len(ids) > 1}
    repeat_agreement: dict[str, Any] = {}
    for criterion in key["criteria"]:
        matches = 0
        total = 0
        for response in validated:
            answer_map = {answer["trial_id"]: answer for answer in response["answers"]}
            for trial_ids in repeated.values():
                first, second = trial_ids[:2]
                first_choice = answer_map[first]["ratings"][criterion]
                second_choice = answer_map[second]["ratings"][criterion]
                first_winner = _winner(first_choice, trials[first])
                second_winner = _winner(second_choice, trials[second])
                matches += first_winner == second_winner
                total += 1
        repeat_agreement[criterion] = {
            "matches": matches,
            "comparisons": total,
            "agreement_rate": round(matches / total, 6) if total else None,
        }
    strata_trials: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for trial_id, trial in trials.items():
        for dimension, value in trial.get("labels", {}).items():
            strata_trials[str(dimension)][str(value)].add(trial_id)
    strata: dict[str, dict[str, Any]] = {}
    for dimension, values in sorted(strata_trials.items()):
        strata[dimension] = {}
        for value, stratum_ids in sorted(values.items()):
            subset = {trial_id: trials[trial_id] for trial_id in stratum_ids}
            strata[dimension][value] = {
                "trials": len(subset),
                **_rank_trials(subset, validated, key["criteria"]),
            }
    return {
        "schema_version": 1,
        "study_id": key["study_id"],
        "title": key["title"],
        "raters": len(validated),
        "trials_per_rater": len(trials),
        **aggregate,
        "bootstrap": bootstrap,
        "repeat_reliability": repeat_agreement,
        "inter_rater_agreement": _inter_rater_agreement(
            trials, validated, key["criteria"]
        ),
        "side_choice_diagnostics": _side_choice_diagnostics(validated, key["criteria"]),
        "strata": strata,
    }
