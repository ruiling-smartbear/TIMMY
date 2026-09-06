from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import pytest

from music_eval.listening import (
    DEFAULT_CRITERIA,
    _fit_bradley_terry,
    analyze_listening_responses,
    build_listening_study,
    load_comparisons,
    validate_response,
)
from music_eval.listening_report import write_listening_report
from music_eval.listening_web import write_listening_interface


def _manifest(tmp_path: Path, write_wav) -> Path:
    write_wav(tmp_path / "model-alpha.wav", np.zeros(8000))
    write_wav(tmp_path / "model-beta.wav", np.ones(8000) * 0.1)
    manifest = tmp_path / "comparisons.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "rock-1",
                "prompt": "A raw garage-rock chorus",
                "audio_a": "model-alpha.wav",
                "system_a": "secret-alpha",
                "audio_b": "model-beta.wav",
                "system_b": "secret-beta",
                "labels": {"genre": "rock"},
            }
        )
        + "\n"
    )
    return manifest


def _response(key: dict, winning_system: str) -> dict:
    answers = []
    for trial in key["trials"]:
        choice = "A" if trial["system_a"] == winning_system else "B"
        answers.append(
            {
                "trial_id": trial["id"],
                "ratings": {criterion: choice for criterion in key["criteria"]},
                "note": "clear preference",
            }
        )
    return {
        "study_id": key["study_id"],
        "rater_id": "listener-1",
        "answers": answers,
    }


def _key(trials: list[tuple[str, str, str]], criteria: tuple[str, ...] = ("overall",)) -> dict:
    return {
        "study_id": "study",
        "title": "Synthetic",
        "criteria": list(criteria),
        "trials": [
            {"id": trial_id, "source_id": trial_id, "system_a": left, "system_b": right}
            for trial_id, left, right in trials
        ],
    }


def _rater(key: dict, rater_id: str, choices: dict[str, str]) -> dict:
    return {
        "study_id": key["study_id"],
        "rater_id": rater_id,
        "answers": [
            {
                "trial_id": trial["id"],
                "ratings": {criterion: choices[trial["id"]] for criterion in key["criteria"]},
            }
            for trial in key["trials"]
        ],
    }


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    rows = [row[:] + [vector[index]] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(rows[row][column]))
        rows[column], rows[pivot] = rows[pivot], rows[column]
        for row in range(column + 1, size):
            factor = rows[row][column] / rows[column][column]
            for cell in range(column, size + 1):
                rows[row][cell] -= factor * rows[column][cell]
    solution = [0.0] * size
    for row in range(size - 1, -1, -1):
        partial = sum(rows[row][cell] * solution[cell] for cell in range(row + 1, size))
        solution[row] = (rows[row][size] - partial) / rows[row][row]
    return solution


def _reference_bradley_terry(
    count: int, observations: list[tuple[int, int, float]]
) -> list[float]:
    """Observation-by-observation Newton iteration, kept as a pure-Python oracle."""
    theta = [0.0] * count
    ridge = 1e-6
    for _ in range(100):
        gradient = [-ridge * value for value in theta]
        information = [[ridge * (row == col) for col in range(count)] for row in range(count)]
        for left, right, outcome in observations:
            difference = min(30.0, max(-30.0, theta[left] - theta[right]))
            probability = 1.0 / (1.0 + math.exp(-difference))
            residual = outcome - probability
            weight = probability * (1.0 - probability)
            gradient[left] += residual
            gradient[right] -= residual
            information[left][left] += weight
            information[right][right] += weight
            information[left][right] -= weight
            information[right][left] -= weight
        delta = _solve([[cell + 1.0 / count for cell in row] for row in information], gradient)
        theta = [value + step for value, step in zip(theta, delta, strict=True)]
        mean = sum(theta) / count
        theta = [value - mean for value in theta]
        if max(abs(step) for step in delta) < 1e-9:
            break
    return theta


def test_vectorized_bradley_terry_matches_pure_python_reference():
    rng = random.Random(20260906)
    systems = [f"system-{index}" for index in range(6)]
    strengths = [rng.uniform(-1.5, 1.5) for _ in systems]
    observations = []
    for _ in range(240):
        left, right = rng.sample(range(len(systems)), 2)
        if rng.random() < 0.2:
            outcome = 0.5
        else:
            edge = 1.0 / (1.0 + math.exp(strengths[right] - strengths[left]))
            outcome = 1.0 if rng.random() < edge else 0.0
        observations.append((left, right, outcome))

    rows = _fit_bradley_terry(systems, observations)
    reference = _reference_bradley_terry(len(systems), observations)

    expected_ranks = {
        value: rank
        for rank, value in enumerate(sorted({round(v, 6) for v in reference}, reverse=True), 1)
    }
    for index, row in enumerate(rows):
        assert row["system"] == systems[index]
        assert row["log_strength"] == pytest.approx(reference[index], abs=1e-6)
        assert row["vs_average"] == pytest.approx(
            1.0 / (1.0 + math.exp(-reference[index])), abs=1e-6
        )
        assert row["rank"] == expected_ranks[round(reference[index], 6)]
        assert row["separated"] is False
    assert len(set(reference)) == len(systems)


def test_all_tie_study_shares_rank_one_and_is_not_separated():
    key = _key([("one", "alpha", "beta"), ("two", "beta", "alpha")])
    raters = [
        _rater(key, "listener-1", {"one": "tie", "two": "tie"}),
        _rater(key, "listener-2", {"one": "tie", "two": "tie"}),
    ]

    result = analyze_listening_responses(key, raters, bootstrap_samples=10)

    for row in result["rankings"]["overall"]:
        assert row["rank"] == 1
        assert row["log_strength"] == 0.0
        assert row["vs_average"] == 0.5
        assert row["separated"] is False
        assert row["log_strength_ci95"] == [0.0, 0.0]


def test_single_unanimous_vote_is_flagged_separated(tmp_path):
    key = _key([("one", "alpha", "beta")])

    result = analyze_listening_responses(key, [_rater(key, "listener", {"one": "A"})])

    rows = result["rankings"]["overall"]
    assert [row["system"] for row in rows] == ["alpha", "beta"]
    assert [row["rank"] for row in rows] == [1, 2]
    assert all(row["separated"] is True for row in rows)
    assert rows[0]["log_strength"] > 5.0
    write_listening_report(result, tmp_path)
    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "| 1* | 1 | alpha |" in markdown
    assert "\\* Separated data:" in markdown
    html_report = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "<td>1*</td>" in html_report
    assert "* Separated data:" in html_report


def test_cyclic_preferences_are_not_separated_and_share_ranks():
    key = _key([("ab", "a", "b"), ("bc", "b", "c"), ("ca", "c", "a")])
    choices = {"ab": "A", "bc": "A", "ca": "A"}

    result = analyze_listening_responses(key, [_rater(key, "listener", choices)])

    rows = result["rankings"]["overall"]
    assert [row["rank"] for row in rows] == [1, 1, 1]
    assert all(row["separated"] is False for row in rows)
    assert all(row["log_strength"] == 0.0 for row in rows)


def test_bootstrap_runs_on_vectorized_fit(tmp_path):
    key = _key([("ab", "a", "b"), ("bc", "b", "c"), ("ca", "c", "a")], DEFAULT_CRITERIA[:2])
    raters = [
        _rater(key, "listener-1", {"ab": "A", "bc": "A", "ca": "B"}),
        _rater(key, "listener-2", {"ab": "A", "bc": "tie", "ca": "B"}),
        _rater(key, "listener-3", {"ab": "B", "bc": "A", "ca": "A"}),
    ]

    result = analyze_listening_responses(key, raters, bootstrap_samples=25, bootstrap_seed=3)

    assert result["bootstrap"]["enabled"] is True
    for criterion in DEFAULT_CRITERIA[:2]:
        rows = result["rankings"][criterion]
        assert [row["system"] for row in rows] == ["a", "b", "c"]
        assert [row["rank"] for row in rows] == [1, 2, 3]
        assert all(row["separated"] is False for row in rows)
        for row in rows:
            low, high = row["log_strength_ci95"]
            assert low <= high
            assert len(row["preference_rate_ci95"]) == 2
    write_listening_report(result, tmp_path)
    assert "Separated data" not in (tmp_path / "report.md").read_text(encoding="utf-8")


def test_null_note_normalizes_to_empty_string():
    key = _key([("one", "alpha", "beta")])
    response = _rater(key, "listener", {"one": "A"})
    response["answers"][0]["note"] = None

    validated = validate_response(response, key)

    assert validated["answers"][0]["note"] == ""


def test_non_string_submitted_at_is_rejected():
    key = _key([("one", "alpha", "beta")])
    response = _rater(key, "listener", {"one": "A"})
    response["submitted_at"] = 1725580800

    with pytest.raises(ValueError, match="submitted_at must be a string"):
        validate_response(response, key)


def test_public_study_file_is_rejected_as_organizer_key(tmp_path, write_wav):
    comparisons = load_comparisons(_manifest(tmp_path, write_wav))
    public_path, _, _ = build_listening_study(
        comparisons, tmp_path / "study", title="Blind test", seed=1, repeat_fraction=0
    )
    public = json.loads(public_path.read_text(encoding="utf-8"))

    with pytest.raises(ValueError, match="looks like the public study file"):
        analyze_listening_responses(public, [])


def test_study_separates_public_data_from_organizer_key(tmp_path, write_wav):
    comparisons = load_comparisons(_manifest(tmp_path, write_wav))
    public_path, key_path, count = build_listening_study(
        comparisons,
        tmp_path / "study",
        title="Blind test",
        seed=7,
        repeat_fraction=1.0,
    )

    public_text = public_path.read_text()
    public = json.loads(public_text)
    key = json.loads(key_path.read_text())
    assert count == 2
    assert key_path.parent == tmp_path
    assert "secret-alpha" not in public_text
    assert "secret-beta" not in public_text
    assert "model-alpha" not in public_text
    assert "repeat_of" not in public_text
    assert {trial["source_id"] for trial in key["trials"]} == {"rock-1"}
    assert len(list((tmp_path / "study" / "audio").iterdir())) == 2

    write_listening_interface(public, tmp_path / "study" / "index.html")
    html = (tmp_path / "study" / "index.html").read_text()
    assert "secret-alpha" not in html
    assert "Pairwise human evaluation" not in html
    assert "pairwise human evaluation" in html


def test_repeat_reliability_uses_system_identity_not_left_right(tmp_path, write_wav):
    comparisons = load_comparisons(_manifest(tmp_path, write_wav))
    _, key_path, _ = build_listening_study(
        comparisons,
        tmp_path / "study",
        title="Blind test",
        seed=2,
        repeat_fraction=1.0,
    )
    key = json.loads(key_path.read_text())
    response = _response(key, "secret-beta")

    result = analyze_listening_responses(key, [response])

    for criterion in DEFAULT_CRITERIA:
        assert result["repeat_reliability"][criterion] == {
            "matches": 1,
            "comparisons": 1,
            "agreement_rate": 1.0,
        }
        ranking = result["rankings"][criterion]
        assert ranking[0]["system"] == "secret-beta"
        assert ranking[0]["rank"] == 1
        assert ranking[0]["preference_rate"] == 1.0
    rock = result["strata"]["genre"]["rock"]
    assert rock["trials"] == 2
    assert rock["rankings"]["overall_preference"][0]["system"] == "secret-beta"


def test_response_must_be_complete(tmp_path, write_wav):
    comparisons = load_comparisons(_manifest(tmp_path, write_wav))
    _, key_path, _ = build_listening_study(
        comparisons,
        tmp_path / "study",
        title="Blind test",
        seed=1,
        repeat_fraction=0,
    )
    key = json.loads(key_path.read_text())
    response = _response(key, "secret-alpha")
    response["answers"] = []

    with pytest.raises(ValueError, match="incomplete"):
        validate_response(response, key)


def test_manifest_rejects_missing_audio(tmp_path):
    manifest = tmp_path / "comparisons.jsonl"
    manifest.write_text(
        '{"id":"x","prompt":"p","audio_a":"missing.wav","system_a":"a",'
        '"audio_b":"missing2.wav","system_b":"b"}\n'
    )
    with pytest.raises(ValueError, match="audio file not found"):
        load_comparisons(manifest)


def test_analysis_refuses_global_rank_for_disconnected_systems():
    key = {
        "study_id": "study",
        "title": "Disconnected",
        "criteria": ["overall"],
        "trials": [
            {
                "id": "one",
                "source_id": "one",
                "system_a": "a",
                "system_b": "b",
            },
            {
                "id": "two",
                "source_id": "two",
                "system_a": "c",
                "system_b": "d",
            },
        ],
    }
    response = {
        "study_id": "study",
        "rater_id": "listener",
        "answers": [
            {"trial_id": "one", "ratings": {"overall": "A"}},
            {"trial_id": "two", "ratings": {"overall": "B"}},
        ],
    }

    result = analyze_listening_responses(key, [response])

    assert result["comparison_graph_connected"] is False
    assert result["comparison_components"] == [["a", "b"], ["c", "d"]]
    assert all(row["rank"] is None for row in result["rankings"]["overall"])


def test_analysis_rejects_duplicate_rater_ids(tmp_path, write_wav):
    comparisons = load_comparisons(_manifest(tmp_path, write_wav))
    _, key_path, _ = build_listening_study(
        comparisons,
        tmp_path / "study",
        title="Blind test",
        seed=1,
        repeat_fraction=0,
    )
    key = json.loads(key_path.read_text())
    response = _response(key, "secret-alpha")

    with pytest.raises(ValueError, match="duplicate rater_id"):
        analyze_listening_responses(key, [response, response])


def test_rater_bootstrap_agreement_and_side_diagnostics(tmp_path, write_wav):
    comparisons = load_comparisons(_manifest(tmp_path, write_wav))
    _, key_path, _ = build_listening_study(
        comparisons,
        tmp_path / "study",
        title="Blind test",
        seed=3,
        repeat_fraction=1,
    )
    key = json.loads(key_path.read_text())
    first = _response(key, "secret-beta")
    second = json.loads(json.dumps(first))
    second["rater_id"] = "listener-2"

    result = analyze_listening_responses(
        key, [first, second], bootstrap_samples=40, bootstrap_seed=5
    )

    assert result["bootstrap"] == {
        "enabled": True,
        "unit": "rater",
        "samples": 40,
        "seed": 5,
        "interval": "percentile_95",
        "captures": "rater sampling uncertainty only",
    }
    for criterion in DEFAULT_CRITERIA:
        top = result["rankings"][criterion][0]
        assert top["system"] == "secret-beta"
        assert top["preference_rate_ci95"] == [1.0, 1.0]
        assert len(top["log_strength_ci95"]) == 2
        assert result["inter_rater_agreement"][criterion][
            "pairwise_agreement_rate"
        ] == 1.0
        counts = result["side_choice_diagnostics"][criterion]["counts"]
        assert sum(counts.values()) == 4
    report_dir = tmp_path / "report"
    write_listening_report(result, report_dir)
    markdown = (report_dir / "report.md").read_text()
    assert "40 rater-level bootstrap samples" in markdown
    assert "inter-rater pair agreement" in markdown
    assert "[100.0%, 100.0%]" in markdown


@pytest.mark.parametrize(
    ("preset", "expected"),
    [
        ("musicprefs", ("fidelity", "musicality")),
        (
            "songeval",
            (
                "overall_coherence",
                "memorability",
                "vocal_naturalness",
                "structure_clarity",
                "overall_musicality",
            ),
        ),
    ],
)
def test_criteria_presets_build_studies_with_definitions(tmp_path, write_wav, preset, expected):
    from music_eval.cli import main
    from music_eval.listening import CRITERIA_DEFINITIONS, CRITERIA_PRESETS

    assert CRITERIA_PRESETS[preset] == expected
    assert all(name in CRITERIA_DEFINITIONS for name in expected)
    comparisons = _manifest(tmp_path, write_wav)
    study_dir = tmp_path / f"study-{preset}"

    assert main(
        [
            "init-listening-study",
            str(comparisons),
            "--output",
            str(study_dir),
            "--criteria-preset",
            preset,
        ]
    ) == 0

    public = json.loads((study_dir / "study.json").read_text(encoding="utf-8"))
    assert tuple(public["criteria"]) == expected
    assert set(public["criteria_definitions"]) == set(expected)
    page = (study_dir / "index.html").read_text(encoding="utf-8")
    assert CRITERIA_DEFINITIONS[expected[0]] in page


def test_songeval_instrumental_preset_drops_the_vocal_dimension():
    from music_eval.listening import CRITERIA_PRESETS

    assert "vocal_naturalness" in CRITERIA_PRESETS["songeval"]
    assert "vocal_naturalness" not in CRITERIA_PRESETS["songeval-instrumental"]
    assert len(CRITERIA_PRESETS["songeval-instrumental"]) == 4


def test_criteria_and_preset_are_mutually_exclusive(tmp_path, write_wav, capsys):
    from music_eval.cli import main

    comparisons = _manifest(tmp_path, write_wav)

    with pytest.raises(SystemExit) as exit_info:
        main(
            [
                "init-listening-study",
                str(comparisons),
                "--criteria",
                "a,b",
                "--criteria-preset",
                "musicprefs",
            ]
        )

    assert exit_info.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err
