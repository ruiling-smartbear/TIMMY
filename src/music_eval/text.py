from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version


def markdown_cell(value: object) -> str:
    """Render ``value`` for a Markdown table cell: pipes escaped, no line breaks."""
    return str(value).replace("|", "\\|").replace("\n", " ")


def package_version(distribution: str) -> str:
    """Installed version of ``distribution``, or "unknown" when it is not installed."""
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unknown"
