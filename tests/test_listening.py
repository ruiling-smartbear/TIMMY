from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from music_eval.listening import (
    DEFAULT_CRITERIA,
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
