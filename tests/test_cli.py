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
