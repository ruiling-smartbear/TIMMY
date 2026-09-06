from __future__ import annotations

import json

import numpy as np
import pytest

from music_eval.audio import read_wav
from music_eval.perturbations import prepare_fidelity_perturbations


def test_fidelity_perturbations_are_deterministic_and_ordered(tmp_path, write_wav):
    source = tmp_path / "source.wav"
    time = np.arange(8000) / 8000
    write_wav(source, 0.2 * np.sin(2 * np.pi * 220 * time))

    first = prepare_fidelity_perturbations(
        [source], tmp_path / "first", sigmas=(0.0, 0.05, 0.1), seed=17
    )
    prepare_fidelity_perturbations(
        [source], tmp_path / "second", sigmas=(0.0, 0.05, 0.1), seed=17
    )

    assert first["noise_sigmas"] == [0.0, 0.05, 0.1]
    first_files = sorted((tmp_path / "first/audio").glob("*.wav"))
    second_files = sorted((tmp_path / "second/audio").glob("*.wav"))
    assert [path.read_bytes() for path in first_files] == [
        path.read_bytes() for path in second_files
    ]
    assert np.array_equal(read_wav(source).samples, read_wav(first_files[0]).samples)
    assert not np.array_equal(read_wav(first_files[0]).samples, read_wav(first_files[2]).samples)


def test_fidelity_perturbations_write_self_contained_manifest_and_provenance(
    tmp_path, write_wav
):
    source = tmp_path / "source.wav"
    write_wav(source, np.zeros(100))
    output = tmp_path / "audit"

    result = prepare_fidelity_perturbations([source], output, sigmas=(0.0, 0.2))
    rows = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]

    assert len(rows) == 2
    assert all((output / row["audio"]).is_file() for row in rows)
    assert all((output / row["reference"]).is_file() for row in rows)
    assert result["protocol"] == "mad-fidelity-gaussian-noise-strengths"
    assert json.loads((output / "provenance.json").read_text())["cases"] == result["cases"]


@pytest.mark.parametrize("sigmas", [(), (0.1, 0.0), (0.1, 0.1), (float("nan"),)])
def test_fidelity_perturbations_reject_invalid_level_design(tmp_path, write_wav, sigmas):
    source = tmp_path / "source.wav"
    write_wav(source, np.zeros(100))

    with pytest.raises(ValueError):
        prepare_fidelity_perturbations([source], tmp_path / "out", sigmas=sigmas)
