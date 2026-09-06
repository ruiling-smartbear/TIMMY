from __future__ import annotations

from music_eval.models import EvaluationResult
from music_eval.report import write_html_report, write_markdown_report


def test_reports_surface_optional_learned_metrics(tmp_path):
    result = EvaluationResult(
        id="song",
        audio="song.wav",
        status="pass",
        labels={"genre": ("rock",)},
        metrics={
            "integrity": {
                "duration_seconds": 10.0,
                "rms_dbfs": -14.0,
                "longest_dropout_seconds": 0.0,
            },
            "audiobox_aesthetics": {
                "track_scores": {"CE": 7.0, "CU": 6.0, "PC": 5.0, "PQ": 8.0}
            },
            "clap_alignment": {
                "track": {
                    "positive_similarity": 0.42,
                    "margin": 0.17,
                    "positive_rank": 1,
                }
            },
            "muq_mulan_alignment": {
                "track": {
                    "positive_similarity": 0.51,
                    "margin": 0.23,
                    "positive_rank": 1,
                }
            },
            "temporal_consistency": {
                "first_to_last": {"spectral_similarity": 0.65},
                "nonlocal_repetition": {"near_duplicate_window_ratio": 0.25},
            },
            "production_quality": {
                "loudness": {
                    "integrated_lufs": -14.0,
                    "loudness_range_lu": 5.0,
                    "estimated_true_peak_dbtp": -1.0,
                }
            },
        },
    )
    markdown = tmp_path / "report.md"
    html = tmp_path / "report.html"

    write_markdown_report([result], markdown)
    write_html_report([result], html)

    markdown_text = markdown.read_text()
    html_text = html.read_text()
    assert (
        "| CE | CU | PC | PQ | CLAP+ | Margin | Rank | MuLan+ | MuLan margin | "
        "MuLan rank | First↔Last | Repeat | "
        "LUFS | LRA | Est. dBTP |" in markdown_text
    )
    assert (
        "| 7.00 | 6.00 | 5.00 | 8.00 | 0.420 | 0.170 | 1 | 0.510 | 0.230 | "
        "1 | 0.650 | 25.0% | "
        "-14.00 | 5.00 | -1.00 |"
        in markdown_text
    )
    assert "<th>CE</th>" in html_text
    assert "<th>CLAP+</th>" in html_text
    assert "<th>MuLan+</th>" in html_text
    assert "<th>First↔Last</th>" in html_text
    assert "<th>LUFS</th>" in html_text


def test_markdown_report_renders_absent_alignment_margin_as_dash(tmp_path):
    result = EvaluationResult(
        id="song",
        audio="song.wav",
        status="pass",
        metrics={
            "muq_mulan_alignment": {
                "track": {
                    "positive_similarity": 0.51,
                    "margin": None,
                    "positive_rank": 1,
                }
            }
        },
    )
    markdown = tmp_path / "report.md"

    write_markdown_report([result], markdown)

    text = markdown.read_text()
    assert "| 0.510 | — | 1 |" in text
    assert "None" not in text
