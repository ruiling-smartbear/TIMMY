from __future__ import annotations

from music_eval.text import markdown_cell, package_version


def test_markdown_cell_escapes_pipes_and_newlines():
    assert markdown_cell("cand|x\nnext") == "cand\\|x next"
    assert markdown_cell(3.5) == "3.5"


def test_package_version_reports_unknown_for_missing_distribution():
    assert package_version("music-eval-no-such-distribution") == "unknown"
    assert package_version("numpy") != "unknown"
