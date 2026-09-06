from __future__ import annotations

import json

import numpy as np

from music_eval.cli import main


def test_cli_writes_json_and_markdown_reports(tmp_path, write_wav):
    time = np.arange(8000) / 8000
    write_wav(tmp_path / "song.wav", 0.25 * np.sin(2 * np.pi * 220 * time))
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "song",
                "audio": "song.wav",
                "expectations": {"duration_seconds": 1.0},
            }
        )
        + "\n"
    )
    output = tmp_path / "report"

    exit_code = main(["evaluate", str(manifest), "--output", str(output)])

    assert exit_code == 0
    report = json.loads((output / "report.json").read_text())
    assert report["summary"] == {
        "total": 1,
        "passed": 1,
        "warnings": 0,
        "failed": 0,
        "pass_rate": 1.0,
    }
    assert "| song | — | pass |" in (output / "report.md").read_text()
    assert (output / "report.html").read_text().startswith("<!doctype html>")


def test_cli_returns_nonzero_for_failed_contract(tmp_path, write_wav):
    write_wav(tmp_path / "song.wav", np.zeros(8000))
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"id":"silent","audio":"song.wav"}\n')

    assert main(["evaluate", str(manifest), "--output", str(tmp_path / "out")]) == 1


def test_cli_rejects_invalid_analysis_window(tmp_path, capsys):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"id":"x","audio":"x.wav"}\n')

    exit_code = main(
        ["evaluate", str(manifest), "--window-seconds", "0", "--output", str(tmp_path)]
    )

    assert exit_code == 2
    assert "--window-seconds must be positive" in capsys.readouterr().err


def test_listening_study_cli_builds_and_analyzes(tmp_path, write_wav):
    write_wav(tmp_path / "a.wav", np.zeros(8000))
    write_wav(tmp_path / "b.wav", np.ones(8000) * 0.1)
    comparisons = tmp_path / "comparisons.jsonl"
    comparisons.write_text(
        '{"id":"one","prompt":"gentle tone","audio_a":"a.wav","system_a":"alpha",'
        '"audio_b":"b.wav","system_b":"beta"}\n'
    )
    study_dir = tmp_path / "study"

    assert main(
        [
            "init-listening-study",
            str(comparisons),
            "--output",
            str(study_dir),
            "--repeat-fraction",
            "0",
        ]
    ) == 0
    key_path = tmp_path / "study.organizer.json"
    key = json.loads(key_path.read_text())
    trial = key["trials"][0]
    response = {
        "study_id": key["study_id"],
        "rater_id": "rater",
        "answers": [
            {
                "trial_id": trial["id"],
                "ratings": {criterion: "A" for criterion in key["criteria"]},
            }
        ],
    }
    response_path = tmp_path / "response.json"
    response_path.write_text(json.dumps(response))
    report = tmp_path / "human-report"

    assert main(
        [
            "analyze-listening-study",
            str(key_path),
            str(response_path),
            "--output",
            str(report),
        ]
    ) == 0
    assert (study_dir / "index.html").is_file()
    assert (report / "report.json").is_file()
    assert (report / "report.md").is_file()
    assert (report / "report.html").is_file()


def test_distribution_cli_writes_reports(tmp_path):
    manifest = tmp_path / "embeddings.jsonl"
    rows = []
    for system in ("reference", "candidate"):
        for index, embedding in enumerate(([1.0, 0.0], [0.0, 1.0])):
            rows.append(
                {
                    "id": f"{system}-{index}",
                    "system": system,
                    "embedding": embedding,
                    "embedding_model": "fixture",
                    "checkpoint": "v1",
                    "labels": {"genre": "test"},
                }
            )
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "distribution-report"

    assert main(
        [
            "compare-distributions",
            str(manifest),
            "--reference-system",
            "reference",
            "--output",
            str(output),
        ]
    ) == 0
    result = json.loads((output / "report.json").read_text())
    assert result["comparisons"]["candidate"]["frechet_embedding_distance"] == 0
    assert (output / "report.md").is_file()
    assert (output / "report.html").is_file()


def test_metric_ordering_cli_writes_reports(tmp_path):
    manifest = tmp_path / "ordering.jsonl"
    rows = [
        {
            "id": f"noise-{level}",
            "condition": "white-noise",
            "metric": "distance",
            "level": level,
            "score": score,
            "expected_direction": "increase",
        }
        for level, score in ((0.0, 0.1), (0.1, 0.4), (0.2, 0.8))
    ]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "ordering-report"

    assert main(
        ["analyze-metric-ordering", str(manifest), "--output", str(output)]
    ) == 0
    result = json.loads((output / "report.json").read_text())
    assert result["groups"][0]["kendall_tau_b"] == 1.0
    assert (output / "report.md").is_file()
    assert (output / "report.html").is_file()


def test_prepare_fidelity_perturbations_cli_writes_audit_fixture(tmp_path, write_wav):
    source = tmp_path / "source.wav"
    write_wav(source, np.zeros(100))
    output = tmp_path / "perturbations"

    assert main(
        [
            "prepare-fidelity-perturbations",
            str(source),
            "--output",
            str(output),
            "--sigmas",
            "0",
            "0.1",
            "--seed",
            "7",
        ]
    ) == 0
    assert len((output / "manifest.jsonl").read_text().splitlines()) == 2
    assert json.loads((output / "provenance.json").read_text())["seed"] == 7
