from __future__ import annotations

import numpy as np

from music_eval.evaluator import evaluate_entry
from music_eval.models import ManifestEntry
from music_eval.plugins import (
    AnalysisConfig,
    MetricOutput,
    available_plugins,
    resolve_plugins,
)


class ConstantMetric:
    name = "constant"

    def evaluate(self, entry, candidate, reference, config):
        return MetricOutput(metrics={"value": 42})


class BrokenMetric:
    name = "broken"

    def evaluate(self, entry, candidate, reference, config):
        raise RuntimeError("intentional test failure")


def test_custom_metric_can_run_without_changing_core(tmp_path, write_wav):
    path = write_wav(tmp_path / "tone.wav", np.full(8000, 0.1))

    result = evaluate_entry(
        ManifestEntry(id="plugin", audio=path),
        config=AnalysisConfig(),
        plugins=[ConstantMetric()],
    )

    assert result.status == "pass"
    assert result.metrics == {"constant": {"value": 42}}


def test_plugin_exception_becomes_attributed_failure(tmp_path, write_wav):
    path = write_wav(tmp_path / "tone.wav", np.full(8000, 0.1))

    result = evaluate_entry(ManifestEntry(id="broken", audio=path), plugins=[BrokenMetric()])

    assert result.status == "fail"
    assert result.findings[0].source == "broken"
    assert result.findings[0].code == "plugin_error"


def test_builtin_plugins_are_discoverable():
    assert {
        "audiobox_aesthetics",
        "clap_alignment",
        "integrity",
        "muq_mulan_alignment",
        "pairwise",
        "production_quality",
        "temporal_consistency",
    } <= set(available_plugins())


def test_duplicate_plugin_names_are_rejected(tmp_path, write_wav):
    path = write_wav(tmp_path / "tone.wav", np.full(8000, 0.1))

    try:
        evaluate_entry(
            ManifestEntry(id="duplicate", audio=path),
            plugins=[ConstantMetric(), ConstantMetric()],
        )
    except ValueError as error:
        assert "must be unique" in str(error)
    else:
        raise AssertionError("duplicate plugin namespaces should fail")


def test_explicit_empty_plugin_list_runs_no_plugins(tmp_path, write_wav):
    path = write_wav(tmp_path / "tone.wav", np.full(8000, 0.1))

    result = evaluate_entry(ManifestEntry(id="none", audio=path), plugins=[])

    assert result.status == "pass"
    assert result.metrics == {}
    assert result.findings == []
    assert resolve_plugins([]) == []
    assert [plugin.name for plugin in resolve_plugins()] == ["integrity", "pairwise"]
