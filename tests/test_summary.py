from __future__ import annotations

from music_eval.models import EvaluationResult, Finding
from music_eval.summary import grouped_summary


def _result(identifier, status, genre, rms, findings=()):
    return EvaluationResult(
        id=identifier,
        audio=f"{identifier}.wav",
        status=status,
        labels={"genre": (genre,), "mood": ("bright",)},
        metrics={
            "integrity": {
                "duration_seconds": 16.0,
                "rms_dbfs": rms,
                "silent_window_ratio": 0.0,
                "longest_dropout_seconds": 0.0,
            }
        },
        findings=list(findings),
    )


def test_grouped_summary_keeps_genres_separate():
    results = [
        _result("j1", "pass", "j-pop", -18.0),
        _result(
            "j2",
            "fail",
            "j-pop",
            -22.0,
            [Finding("duration_mismatch", "failure", "too short", "integrity")],
        ),
        _result("a1", "pass", "ambient", -30.0),
    ]

    groups = grouped_summary(results)

    assert groups["genre"]["j-pop"]["total"] == 2
    assert groups["genre"]["j-pop"]["pass_rate"] == 0.5
    assert groups["genre"]["j-pop"]["mean_rms_dbfs"] == -20.0
    assert groups["genre"]["ambient"]["pass_rate"] == 1.0
    assert groups["genre"]["j-pop"]["finding_counts"] == {
        "integrity:duration_mismatch": 1
    }


def test_grouped_summary_includes_audiobox_axes():
    result = _result("j1", "pass", "j-pop", -18.0)
    result.metrics["audiobox_aesthetics"] = {
        "track_scores": {"CE": 7.0, "CU": 6.0, "PC": 5.0, "PQ": 8.0}
    }

    groups = grouped_summary([result])

    assert groups["genre"]["j-pop"]["mean_audiobox_ce"] == 7.0
    assert groups["genre"]["j-pop"]["mean_audiobox_pq"] == 8.0


def test_grouped_summary_includes_clap_alignment():
    result = _result("j1", "pass", "j-pop", -18.0)
    result.metrics["clap_alignment"] = {
        "track": {"positive_similarity": 0.42, "margin": 0.17}
    }

    groups = grouped_summary([result])

    assert groups["genre"]["j-pop"]["mean_clap_positive_similarity"] == 0.42
    assert groups["genre"]["j-pop"]["mean_clap_margin"] == 0.17


def test_grouped_summary_includes_temporal_evidence():
    result = _result("j1", "pass", "j-pop", -18.0)
    result.metrics["temporal_consistency"] = {
        "first_to_last": {"spectral_similarity": 0.65},
        "nonlocal_repetition": {"near_duplicate_window_ratio": 0.25},
    }

    groups = grouped_summary([result])

    assert groups["genre"]["j-pop"]["mean_first_to_last_spectral_similarity"] == 0.65
    assert groups["genre"]["j-pop"]["mean_near_duplicate_window_ratio"] == 0.25
