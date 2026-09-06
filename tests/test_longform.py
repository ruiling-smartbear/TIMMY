from __future__ import annotations

import json

import pytest

from music_eval.cli import main
from music_eval.longform import DEFAULT_LONGFORM_DURATIONS, write_longform_manifest
from music_eval.manifest import load_manifest


def test_longform_protocol_keeps_prompt_and_seed_matched(tmp_path):
    output = tmp_path / "longform.jsonl"

    count = write_longform_manifest(
        "generative-music-v1",
        output,
        case_ids=("ambient-drone",),
        seeds=(17,),
    )
    entries = load_manifest(output)

    assert count == len(entries) == 4
    assert {entry.prompt for entry in entries} == {
        "Slow-evolving ambient drone with deep textures, wide space, and no percussion"
    }
    assert {entry.seed for entry in entries} == {17}
    assert tuple(entry.expectations.duration_seconds for entry in entries) == (
        DEFAULT_LONGFORM_DURATIONS
    )
    assert len({entry.labels["longform_group"] for entry in entries}) == 1
    assert len({entry.audio for entry in entries}) == 4


def test_longform_cli_can_select_cases_and_durations(tmp_path):
    output = tmp_path / "longform.jsonl"

    assert main(
        [
            "init-longform-study",
            str(output),
            "--case-ids",
            "rock-arena",
            "--durations",
            "30",
            "120",
            "--seeds",
            "3",
        ]
    ) == 0

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 2
    assert [row["expectations"]["duration_seconds"] for row in rows] == [30.0, 120.0]
    assert [row["labels"]["target_duration_seconds"] for row in rows] == ["30", "120"]


def test_longform_writes_optional_format_contract_and_fractional_slug(tmp_path):
    output = tmp_path / "longform.jsonl"

    write_longform_manifest(
        "generative-music-v1",
        output,
        durations_seconds=(30.5,),
        seeds=(9,),
        case_ids=("jpop-bright-band",),
        audio_directory="generated/",
        sample_rate=48_000,
        channels=2,
    )
    row = json.loads(output.read_text())

    assert row["id"].endswith("-d30p5s")
    assert row["audio"] == "generated/jpop-bright-band-seed9-d30p5s.wav"
    assert row["expectations"]["sample_rate"] == 48_000
    assert row["expectations"]["channels"] == 2


def test_longform_default_selects_the_entire_suite(tmp_path):
    output = tmp_path / "longform.jsonl"

    count = write_longform_manifest(
        "generative-music-v1", output, durations_seconds=(30.0,), seeds=(1,)
    )

    assert count == 16


def test_longform_overwrite_is_explicit(tmp_path):
    output = tmp_path / "longform.jsonl"
    output.write_text("do not erase")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_longform_manifest("generative-music-v1", output)
    assert output.read_text() == "do not erase"

    assert write_longform_manifest(
        "generative-music-v1",
        output,
        durations_seconds=(30.0,),
        seeds=(1,),
        case_ids=("ambient-drone",),
        overwrite=True,
    ) == 1


@pytest.mark.parametrize(
    ("durations", "message"),
    [
        ((30.0, 30.0), "unique"),
        ((120.0, 30.0), "increasing"),
        ((30.0, float("nan")), "positive finite"),
    ],
)
def test_longform_rejects_invalid_duration_design(tmp_path, durations, message):
    with pytest.raises(ValueError, match=message):
        write_longform_manifest(
            "generative-music-v1",
            tmp_path / "longform.jsonl",
            durations_seconds=durations,
            case_ids=("ambient-drone",),
        )


def test_longform_rejects_unknown_case_before_writing(tmp_path):
    output = tmp_path / "longform.jsonl"

    with pytest.raises(ValueError, match="unknown case id"):
        write_longform_manifest(
            "generative-music-v1", output, case_ids=("does-not-exist",)
        )

    assert not output.exists()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"suite_name": "missing"}, "unknown suite"),
        ({"durations_seconds": ()}, "at least one duration"),
        ({"seeds": ()}, "at least one seed"),
        ({"seeds": (9, 9)}, "seeds must be unique"),
        ({"audio_directory": "  "}, "audio directory must be nonempty"),
        ({"sample_rate": 0}, "sample_rate must be positive"),
        ({"channels": 0}, "channels must be positive"),
        ({"case_ids": ()}, "at least one suite case"),
        ({"case_ids": ("ambient-drone", "ambient-drone")}, "case_ids must be unique"),
    ],
)
def test_longform_rejects_invalid_protocol_parameters(tmp_path, kwargs, message):
    arguments = {
        "suite_name": "generative-music-v1",
        "output": tmp_path / "longform.jsonl",
        "durations_seconds": (30.0,),
        "seeds": (9,),
        "case_ids": ("ambient-drone",),
    }
    arguments.update(kwargs)

    with pytest.raises((ValueError, FileExistsError), match=message):
        write_longform_manifest(**arguments)
