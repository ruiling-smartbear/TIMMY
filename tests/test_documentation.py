from __future__ import annotations

import re
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _markdown_files() -> list[Path]:
    return [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]


def test_all_local_markdown_links_resolve():
    missing = []
    for document in _markdown_files():
        for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", document.read_text()):
            if target.startswith(("http://", "https://", "#")):
                continue
            path = target.split("#", 1)[0]
            if path and not (document.parent / path).resolve().exists():
                missing.append(f"{document.relative_to(ROOT)} -> {target}")
    assert not missing, "missing local documentation links:\n" + "\n".join(missing)


def test_readme_names_every_builtin_metric():
    from music_eval.plugins.registry import BUILTINS

    readme = (ROOT / "README.md").read_text()
    missing = [name for name in BUILTINS if name not in readme]
    assert not missing, f"README does not name built-in metrics: {missing}"


def test_rfc_does_not_mark_known_research_gaps_as_implemented():
    rfc = (ROOT / "docs/rfcs/0001-timmy-evidence-framework.md").read_text()
    for claim in ("vocal intelligibility", "local edit success", "workflow/agency"):
        row = next(line for line in rfc.splitlines() if line.startswith(f"| {claim} |"))
        assert row.endswith("| proposed |")


def test_package_versions_stay_in_sync():
    import music_eval

    pyproject = (ROOT / "pyproject.toml").read_text()
    assert f'version = "{music_eval.__version__}"' in pyproject
    citation = (ROOT / "CITATION.cff").read_text()
    assert f"version: {music_eval.__version__}" in citation


def test_timmy_logo_is_horizontal_and_has_real_alpha():
    logo = (ROOT / "assets/timmy-logo-horizontal.png").read_bytes()
    assert logo[:8] == b"\x89PNG\r\n\x1a\n"
    width, height, _depth, color_type = struct.unpack(">IIBB", logo[16:26])
    assert width >= height * 2
    assert color_type in {4, 6}, "logo must use a PNG color type with alpha"
