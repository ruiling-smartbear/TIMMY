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


def _png_chunk_types(data: bytes) -> list[bytes]:
    types, offset = [], 8
    while offset + 8 <= len(data):
        length, chunk_type = struct.unpack(">I4s", data[offset : offset + 8])
        types.append(chunk_type)
        offset += 12 + length
    return types


def test_timmy_logo_is_horizontal_small_and_has_real_alpha():
    logo = (ROOT / "assets/timmy-logo-horizontal.png").read_bytes()
    assert logo[:8] == b"\x89PNG\r\n\x1a\n"
    width, height, _depth, color_type = struct.unpack(">IIBB", logo[16:26])
    assert width >= height * 2
    # Truecolor/gray with alpha, or a palette with a transparency chunk.
    has_alpha = color_type in {4, 6} or (color_type == 3 and b"tRNS" in _png_chunk_types(logo))
    assert has_alpha, "logo must carry alpha so it sits on light and dark backgrounds"
    assert len(logo) < 200_000, "keep the logo small; it is downloaded with every clone"
