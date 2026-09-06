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
        },
    )
    markdown = tmp_path / "report.md"
    html = tmp_path / "report.html"

    write_markdown_report([result], markdown)
    write_html_report([result], html)

    markdown_text = markdown.read_text()
    html_text = html.read_text()
    assert "| CE | CU | PC | PQ | CLAP+ | Margin | Rank |" in markdown_text
    assert "| 7.00 | 6.00 | 5.00 | 8.00 | 0.420 | 0.170 | 1 |" in markdown_text
    assert "<th>CE</th>" in html_text
    assert "<th>CLAP+</th>" in html_text
