from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

from music_eval.models import EvaluationResult
from music_eval.summary import report_payload


def write_json_report(results: list[EvaluationResult], path: Path) -> None:
    payload = report_payload(results)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _integrity(result: EvaluationResult, key: str, default: Any = "—") -> Any:
    metrics = result.metrics.get("integrity")
    return metrics.get(key, default) if isinstance(metrics, dict) else default


def _aesthetic(result: EvaluationResult, axis: str, default: Any = "—") -> Any:
    metrics = result.metrics.get("audiobox_aesthetics")
    if not isinstance(metrics, dict):
        return default
    scores = metrics.get("track_scores")
    return scores.get(axis, default) if isinstance(scores, dict) else default


def write_markdown_report(results: list[EvaluationResult], path: Path) -> None:
    payload = report_payload(results)
    summary = payload["summary"]
    aesthetic_axes = (
        ("CE", "CU", "PC", "PQ")
        if any("audiobox_aesthetics" in result.metrics for result in results)
        else ()
    )
    aesthetic_headers = "".join(f" {axis} |" for axis in aesthetic_axes)
    aesthetic_rules = "".join("---:|" for _axis in aesthetic_axes)
    lines = [
        "# Music evaluation report",
        "",
        f"Total: **{summary['total']}** · Passed: **{summary['passed']}** · "
        f"Warnings: **{summary['warnings']}** · Failed: **{summary['failed']}**",
        "",
        f"| ID | Genre | Status | Duration | RMS | Longest dropout |{aesthetic_headers} Findings |",
        f"|---|---|---:|---:|---:|---:|{aesthetic_rules}---|",
    ]
    for result in results:
        duration = _integrity(result, "duration_seconds")
        rms = _integrity(result, "rms_dbfs")
        longest = _integrity(result, "longest_dropout_seconds")
        genres = ", ".join(result.labels.get("genre", ())) or "—"
        findings = "; ".join(
            f"{finding.source}:{finding.code}" for finding in result.findings
        ) or "—"
        aesthetic_cells = "".join(
            f" {value if value == '—' else f'{float(value):.2f}'} |"
            for axis in aesthetic_axes
            for value in [_aesthetic(result, axis)]
        )
        lines.append(
            f"| {_cell(result.id)} | {_cell(genres)} | {result.status} | "
            f"{duration if duration == '—' else f'{float(duration):.3f}s'} | "
            f"{rms if rms == '—' else f'{float(rms):.2f} dBFS'} | "
            f"{longest if longest == '—' else f'{float(longest):.3f}s'} |"
            f"{aesthetic_cells} "
            f"{_cell(findings)} |"
        )

    groups = payload["groups"]
    for dimension, members in groups.items():
        lines.extend(
            [
                "",
                f"## By {dimension}",
                "",
                "| Value | Samples | Pass | Warning | Fail | Pass rate |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for value, group in members.items():
            lines.append(
                f"| {_cell(value)} | {group['total']} | {group['passed']} | "
                f"{group['warnings']} | {group['failed']} | {group['pass_rate']:.1%} |"
            )

    for result in results:
        if not result.findings:
            continue
        lines.extend(["", f"## {result.id}", ""])
        for finding in result.findings:
            lines.append(
                f"- **{finding.severity} · {finding.source}:{finding.code}:** "
                f"{finding.message}"
            )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_html_report(results: list[EvaluationResult], path: Path) -> None:
    payload = report_payload(results)
    summary = payload["summary"]
    aesthetic_axes = (
        ("CE", "CU", "PC", "PQ")
        if any("audiobox_aesthetics" in result.metrics for result in results)
        else ()
    )
    aesthetic_headers = "".join(f"<th>{axis}</th>" for axis in aesthetic_axes)
    sample_rows = []
    for result in results:
        findings = ", ".join(
            f"{finding.source}:{finding.code}" for finding in result.findings
        ) or "—"
        aesthetic_cells = "".join(
            f"<td>{_html_number(_aesthetic(result, axis), '')}</td>"
            for axis in aesthetic_axes
        )
        sample_rows.append(
            "<tr>"
            f"<td><code>{escape(result.id)}</code></td>"
            f"<td>{escape(', '.join(result.labels.get('genre', ())) or '—')}</td>"
            f"<td><span class='status {result.status}'>{result.status}</span></td>"
            f"<td>{_html_number(_integrity(result, 'duration_seconds'), 's')}</td>"
            f"<td>{_html_number(_integrity(result, 'rms_dbfs'), ' dBFS')}</td>"
            f"<td>{_html_number(_integrity(result, 'longest_dropout_seconds'), 's')}</td>"
            f"{aesthetic_cells}"
            f"<td>{escape(findings)}</td>"
            "</tr>"
        )

    group_sections = []
    for dimension, members in payload["groups"].items():
        rows = "".join(
            "<tr>"
            f"<td>{escape(value)}</td><td>{group['total']}</td>"
            f"<td>{group['passed']}</td><td>{group['warnings']}</td>"
            f"<td>{group['failed']}</td><td>{group['pass_rate']:.1%}</td>"
            "</tr>"
            for value, group in members.items()
        )
        group_sections.append(
            f"<section><h2>By {escape(dimension)}</h2><div class='table-wrap'>"
            "<table><thead><tr><th>Value</th><th>Samples</th><th>Pass</th>"
            "<th>Warning</th><th>Fail</th><th>Pass rate</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div></section>"
        )

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Music evaluation report</title>
<style>
:root{{--ink:#17202a;--muted:#64748b;--line:#e2e8f0;--paper:#fff;--bg:#f6f7f9;
--pass:#177245;--warn:#9a6700;--fail:#b42318}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.5 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{max-width:1180px;margin:auto;padding:48px 24px 80px}} h1{{font-size:34px;margin:0 0 8px}}
h2{{margin:36px 0 12px;font-size:20px}} .subtitle{{color:var(--muted);margin:0 0 28px}}
.cards{{display:grid;grid-template-columns:repeat(4,minmax(120px,1fr));gap:12px}}
.card,section{{background:var(--paper);border:1px solid var(--line);border-radius:14px}}
.card{{padding:18px}} .card strong{{display:block;font-size:28px}} .card span{{color:var(--muted)}}
section{{padding:20px;margin-top:18px}} .table-wrap{{overflow:auto}} table{{width:100%;
border-collapse:collapse}} th,td{{padding:10px 12px;text-align:left;
border-bottom:1px solid var(--line)}}
th{{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}}
.status{{font-weight:700}} .status.pass{{color:var(--pass)}} .status.warning{{color:var(--warn)}}
.status.fail{{color:var(--fail)}} code{{font-size:12px}}
@media(max-width:650px){{.cards{{grid-template-columns:1fr 1fr}}}}
</style>
</head>
<body><main>
<h1>Music evaluation report</h1>
<p class="subtitle">Technical integrity and explicit policy results.
No universal aesthetic score.</p>
<div class="cards">
<div class="card"><strong>{summary['total']}</strong><span>Total</span></div>
<div class="card"><strong>{summary['passed']}</strong><span>Passed</span></div>
<div class="card"><strong>{summary['warnings']}</strong><span>Warnings</span></div>
<div class="card"><strong>{summary['failed']}</strong><span>Failed</span></div>
</div>
<section><h2>Samples</h2><div class="table-wrap"><table><thead><tr><th>ID</th><th>Genre</th>
<th>Status</th><th>Duration</th><th>RMS</th><th>Longest dropout</th>
{aesthetic_headers}<th>Findings</th>
</tr></thead><tbody>{''.join(sample_rows)}</tbody></table></div></section>
{''.join(group_sections)}
</main></body></html>
"""
    path.write_text(document, encoding="utf-8")


def _html_number(value: Any, suffix: str) -> str:
    if not isinstance(value, (int, float)):
        return "—"
    return f"{value:.3f}{suffix}"


def write_reports(results: list[EvaluationResult], output_directory: Path) -> None:
    write_json_report(results, output_directory / "report.json")
    write_markdown_report(results, output_directory / "report.md")
    write_html_report(results, output_directory / "report.html")
