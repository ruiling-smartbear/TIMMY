from __future__ import annotations

from collections import Counter

import pytest

from music_eval.cli import main
from music_eval.manifest import load_manifest
from music_eval.suites import write_suite_manifest


def test_standard_suite_generates_32_stratified_cases(tmp_path):
    output = tmp_path / "suite.jsonl"

    count = write_suite_manifest(
        "generative-music-v1",
        output,
        seeds=(9, 17),
        audio_directory="outputs",
        duration_seconds=16.0,
        sample_rate=32000,
        channels=2,
    )
    entries = load_manifest(output)

    assert count == len(entries) == 32
    genre_counts = Counter(entry.labels["genre"][0] for entry in entries)
    assert len(genre_counts) == 8
    assert set(genre_counts.values()) == {4}
    assert all(len(entry.negative_prompts) == 3 for entry in entries)
    assert all(entry.expectations.duration_seconds == 16.0 for entry in entries)


def test_suite_commands_are_exposed(tmp_path, capsys):
    output = tmp_path / "suite.jsonl"

    assert main(["list-suites"]) == 0
    assert "generative-music-v1" in capsys.readouterr().out
    assert main(["init-suite", str(output), "--seeds", "3"]) == 0
    assert len(load_manifest(output)) == 16


def test_suite_refuses_accidental_overwrite(tmp_path):
    output = tmp_path / "suite.jsonl"
    output.write_text("keep me")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_suite_manifest(
            "generative-music-v1",
            output,
            seeds=(9, 17),
            audio_directory="outputs",
            duration_seconds=16.0,
            sample_rate=None,
            channels=None,
        )

    assert output.read_text() == "keep me"


def test_suite_rejects_duplicate_seeds(tmp_path):
    with pytest.raises(ValueError, match="seeds must be unique"):
        write_suite_manifest(
            "generative-music-v1",
            tmp_path / "suite.jsonl",
            seeds=(9, 9),
            audio_directory="outputs",
            duration_seconds=16.0,
            sample_rate=None,
            channels=None,
        )
