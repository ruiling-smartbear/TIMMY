from __future__ import annotations

from pathlib import Path
from subprocess import CompletedProcess

import numpy as np

from music_eval.audio import AudioData
from music_eval.models import ManifestEntry
from music_eval.plugins.base import AnalysisConfig
from music_eval.plugins.musecp_preservation import (
    MuseCPPreservationMetric,
    _run_structural_score,
)


def _audio() -> AudioData:
    samples = np.zeros((100, 1), dtype=np.float64)
    samples.setflags(write=False)
    return AudioData(samples, 100, 1, 2)


def _functions(calls: list[tuple[str, str, str]]):
    groups = (
        "harmony_tonality",
        "rhythm_meter",
        "structural_form",
        "melodic_content",
        "timbre_texture",
    )

    def make(group: str):
        def score(original: str, edited: str):
            calls.append((group, original, edited))
            return {"similarity": 0.75, "details": [1, 2]}

        return score

    return {group: make(group) for group in groups}


def test_musecp_runs_all_facets_in_original_then_edited_order():
    calls: list[tuple[str, str, str]] = []
    metric = MuseCPPreservationMetric(function_loader=lambda: _functions(calls))
    entry = ManifestEntry(
        id="edit", audio=Path("edited.wav"), reference=Path("original.wav")
    )

    output = metric.evaluate(entry, _audio(), _audio(), AnalysisConfig())

    assert output.metrics["available"] is True
    assert set(output.metrics["facets"]) == {
        "harmony_tonality",
        "rhythm_meter",
        "structural_form",
        "melodic_content",
        "timbre_texture",
    }
    assert all(call[1:] == ("original.wav", "edited.wav") for call in calls)
    assert len(calls) == 5


def test_musecp_without_reference_is_visible_but_does_not_load_dependency():
    loaded = False

    def loader():
        nonlocal loaded
        loaded = True
        return _functions([])

    metric = MuseCPPreservationMetric(function_loader=loader)
    output = metric.evaluate(
        ManifestEntry(id="song", audio=Path("song.wav")),
        _audio(),
        None,
        AnalysisConfig(),
    )

    assert loaded is False
    assert output.metrics == {"available": False, "reason": "reference_required"}
    assert output.findings[0].code == "reference_required"


def test_musecp_rejects_non_finite_third_party_output():
    functions = _functions([])
    functions["rhythm_meter"] = lambda _original, _edited: {"tempo": float("nan")}
    metric = MuseCPPreservationMetric(function_loader=lambda: functions)

    try:
        metric.evaluate(
            ManifestEntry(
                id="edit", audio=Path("edited.wav"), reference=Path("original.wav")
            ),
            _audio(),
            _audio(),
            AnalysisConfig(),
        )
    except ValueError as error:
        assert "non-finite" in str(error)
    else:
        raise AssertionError("non-finite third-party output must fail")


def test_musecp_rejects_missing_or_extra_facet_functions():
    functions = _functions([])
    del functions["structural_form"]
    metric = MuseCPPreservationMetric(function_loader=lambda: functions)

    with np.testing.assert_raises_regex(ValueError, "all five metric families"):
        metric.evaluate(
            ManifestEntry(
                id="edit", audio=Path("edited.wav"), reference=Path("original.wav")
            ),
            _audio(),
            _audio(),
            AnalysisConfig(),
        )


def test_musecp_rejects_unsupported_third_party_value():
    functions = _functions([])
    functions["rhythm_meter"] = lambda _original, _edited: {"tempo": object()}
    metric = MuseCPPreservationMetric(function_loader=lambda: functions)

    with np.testing.assert_raises_regex(ValueError, "unsupported value"):
        metric.evaluate(
            ManifestEntry(
                id="edit", audio=Path("edited.wav"), reference=Path("original.wav")
            ),
            _audio(),
            _audio(),
            AnalysisConfig(),
        )


def test_structural_score_runs_in_an_isolated_temporary_directory(monkeypatch, tmp_path):
    observed: dict[str, object] = {}
    original = tmp_path / "original.wav"
    edited = tmp_path / "edited.wav"
    original.write_bytes(b"original")
    edited.write_bytes(b"edited")

    def fake_run(args, *, cwd, check, stdout, text):
        observed["args"] = args
        observed["cwd_exists_during_call"] = Path(cwd).is_dir()
        observed["cwd"] = cwd
        observed["check"] = check
        observed["text"] = text
        observed["original_copy"] = Path(args[-2]).read_bytes()
        observed["edited_copy"] = Path(args[-1]).read_bytes()
        return CompletedProcess(args, 0, stdout='{"pairwise_f": 1.0, "ari": 1.0}\n')

    monkeypatch.setattr("music_eval.plugins.musecp_preservation.subprocess.run", fake_run)

    assert _run_structural_score(str(original), str(edited)) == {
        "pairwise_f": 1.0,
        "ari": 1.0,
    }
    assert observed["cwd_exists_during_call"] is True
    assert not Path(str(observed["cwd"])).exists()
    assert observed["original_copy"] == b"original"
    assert observed["edited_copy"] == b"edited"
    assert not Path(str(observed["args"][-2])).exists()


def test_structural_score_surfaces_subprocess_failure(monkeypatch, tmp_path):
    original = tmp_path / "original.wav"
    edited = tmp_path / "edited.wav"
    original.write_bytes(b"original")
    edited.write_bytes(b"edited")

    def fake_run(args, **_kwargs):
        return CompletedProcess(args, 7, stdout="")

    monkeypatch.setattr("music_eval.plugins.musecp_preservation.subprocess.run", fake_run)

    with np.testing.assert_raises_regex(RuntimeError, "exited 7"):
        _run_structural_score(str(original), str(edited))


def test_structural_score_rejects_non_json_stdout(monkeypatch, tmp_path):
    original = tmp_path / "original.wav"
    edited = tmp_path / "edited.wav"
    original.write_bytes(b"original")
    edited.write_bytes(b"edited")

    def fake_run(args, **_kwargs):
        return CompletedProcess(args, 0, stdout="warning instead of JSON")

    monkeypatch.setattr("music_eval.plugins.musecp_preservation.subprocess.run", fake_run)

    with np.testing.assert_raises_regex(ValueError, "invalid JSON"):
        _run_structural_score(str(original), str(edited))
