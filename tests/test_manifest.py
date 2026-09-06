from __future__ import annotations

import json

import pytest

from music_eval.manifest import load_manifest


def test_loads_relative_paths_and_expectations(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "song-1",
                "audio": "audio/song.wav",
                "prompt": "bright synthwave",
                "negative_prompts": ["acoustic blues", "ambient drone"],
                "seed": 9,
                "reference": "reference/song.wav",
                "labels": {
                    "Genre": "Synthwave",
                    "Mood": ["Dark", "Driving", "Dark"],
                },
                "expectations": {"sample_rate": 32000, "channels": 2},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    entry = load_manifest(manifest)[0]

    assert entry.audio == tmp_path / "audio/song.wav"
    assert entry.prompt == "bright synthwave"
    assert entry.seed == 9
    assert entry.reference == tmp_path / "reference/song.wav"
    assert entry.negative_prompts == ("acoustic blues", "ambient drone")
    assert entry.labels == {
        "genre": ("synthwave",),
        "mood": ("dark", "driving"),
    }
    assert entry.expectations.sample_rate == 32000


def test_rejects_unknown_expectation(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        '{"id":"x","audio":"x.wav","expectations":{"magic_score":1}}\n'
    )

    with pytest.raises(ValueError, match="unknown expectation"):
        load_manifest(manifest)


def test_rejects_duplicate_ids(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        '{"id":"x","audio":"a.wav"}\n{"id":"x","audio":"b.wav"}\n'
    )

    with pytest.raises(ValueError, match="duplicate id"):
        load_manifest(manifest)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("duration_tolerance_seconds", -1),
        ("sample_rate", 0),
        ("channels", 0),
        ("max_clipped_ratio", 1.1),
        ("max_silent_window_ratio", -0.1),
        ("max_dropout_seconds", -1),
    ],
)
def test_rejects_invalid_expectation_values(tmp_path, field, value):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {"id": "x", "audio": "x.wav", "expectations": {field: value}}
        )
        + "\n"
    )

    with pytest.raises(ValueError, match=field):
        load_manifest(manifest)


def test_rejects_null_duration_tolerance_naming_field_and_row(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        '{"id":"song-7","audio":"x.wav","expectations":'
        '{"duration_seconds":2.0,"duration_tolerance_seconds":null}}\n'
    )

    with pytest.raises(ValueError, match="duration_tolerance_seconds.*song-7"):
        load_manifest(manifest)


def test_rejects_unknown_entry_field_to_prevent_silent_policy_typos(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        '{"id":"x","audio":"x.wav","expectation":{"sample_rate":32000}}\n'
    )

    with pytest.raises(ValueError, match="unknown entry field.*expectation"):
        load_manifest(manifest)


@pytest.mark.parametrize(
    "entry",
    [
        {"id": 1, "audio": "x.wav"},
        {"id": "x", "audio": 1},
        {"id": "x", "audio": "x.wav", "prompt": 1},
        {"id": "x", "audio": "x.wav", "seed": True},
    ],
)
def test_rejects_invalid_entry_types(tmp_path, entry):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(entry) + "\n")

    with pytest.raises(ValueError):
        load_manifest(manifest)


def test_rejects_invalid_labels(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        '{"id":"x","audio":"x.wav","labels":{"genre":["rock",1]}}\n'
    )

    with pytest.raises(ValueError, match="label 'genre'"):
        load_manifest(manifest)
