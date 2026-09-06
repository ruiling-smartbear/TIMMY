# ruff: noqa: E501
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def _number(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def write_meta_evaluation_report(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Metric ordering meta-evaluation",
        "",
        "> This report tests whether a metric follows a known perturbation order. It does not establish that the metric measures music quality in general.",
        "",
        "| Condition | Metric | Expected | Levels | N | Kendall τ-b ↑ | Ordered pairs ↑ | Ties |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in result["groups"]:
        pairs = group["ordered_pairs"]
        lines.append(
            f"| {group['condition']} | {group['metric']} | {group['expected_direction']} | "
            f"{len(group['levels'])} | {group['observations']} | "
            f"{_number(group['kendall_tau_b'])} | "
            f"{_number(group['ordered_pair_accuracy'])} | {pairs['tied']} |"
        )
    for group in result["groups"]:
        lines.extend(
            [
                "",
                f"## {group['condition']} · {group['metric']}",
                "",
                "| Degradation level | N | Mean score | Std. dev. | Minimum | Maximum |",
                "| ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for level in group["levels"]:
            lines.append(
                f"| {level['level']:.6g} | {level['samples']} | "
                f"{level['score_mean']:.6g} | {level['score_std']:.6g} | "
                f"{level['score_min']:.6g} | {level['score_max']:.6g} |"
            )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    rows = []
    for group in result["groups"]:
        rows.append(
            f"<tr><td>{html.escape(group['condition'])}</td>"
            f"<td>{html.escape(group['metric'])}</td>"
            f"<td>{html.escape(group['expected_direction'])}</td>"
            f"<td>{len(group['levels'])}</td><td>{group['observations']}</td>"
            f"<td>{_number(group['kendall_tau_b'])}</td>"
            f"<td>{_number(group['ordered_pair_accuracy'])}</td></tr>"
        )
    document = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Metric meta-evaluation</title><style>body{{margin:0;background:#f5f1e8;color:#172136;font-family:ui-monospace,monospace;padding:52px}}main{{max-width:1050px;margin:auto}}h1{{font:56px Georgia,serif;margin-bottom:10px}}.lede{{max-width:760px;color:#526073}}section{{margin-top:36px;background:#fffaf1;border:1px solid #d8ccba;padding:24px;overflow:auto}}table{{width:100%;border-collapse:collapse}}th,td{{padding:12px;text-align:left;border-bottom:1px solid #ded5c7}}th{{font-size:11px;text-transform:uppercase;color:#7f4561}}</style></head><body><main><h1>Does the metric notice?</h1><p class="lede">Ordered perturbation evidence for metric sensitivity. High agreement is necessary, not sufficient, for perceptual validity.</p><section><table><thead><tr><th>Condition</th><th>Metric</th><th>Expected</th><th>Levels</th><th>N</th><th>Kendall τ-b</th><th>Ordered pairs</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section></main></body></html>"""
    (output_dir / "report.html").write_text(document, encoding="utf-8")
