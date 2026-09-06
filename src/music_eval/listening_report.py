# ruff: noqa: E501
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def write_listening_report(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    lines = [
        f"# {result['title']} — listening results",
        "",
        f"Raters: **{result['raters']}** · trials per rater: **{result['trials_per_rater']}**",
        "",
    ]
    if not result["comparison_graph_connected"]:
        lines.extend(
            [
                "> **No global ranking:** the comparison graph is disconnected. "
                "Strengths are centered separately inside each component.",
                "",
            ]
        )
    cards = []
    for criterion, rows in result["rankings"].items():
        title = criterion.replace("_", " ").title()
        lines.extend(
            [
                f"## {title}",
                "",
                "| Rank | Component | System | Preference rate | BT log strength | vs average | N |",
                "| ---: | ---: | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        body = []
        for row in rows:
            rank = row["rank"] if row["rank"] is not None else "—"
            lines.append(
                f"| {rank} | {row['component']} | {row['system']} | "
                f"{row['preference_rate']:.1%} | "
                f"{row['log_strength']:.3f} | {row['vs_average']:.1%} | {row['comparisons']} |"
            )
            body.append(
                f"<tr><td>{rank}</td><td>{row['component']}</td>"
                f"<td>{html.escape(str(row['system']))}</td>"
                f"<td>{row['preference_rate']:.1%}</td><td>{row['log_strength']:.3f}</td>"
                f"<td>{row['vs_average']:.1%}</td><td>{row['comparisons']}</td></tr>"
            )
        reliability = result["repeat_reliability"][criterion]
        rate = reliability["agreement_rate"]
        rendered_rate = "not measured" if rate is None else f"{rate:.1%}"
        lines.extend(["", f"Repeat-trial agreement: **{rendered_rate}**", ""])
        cards.append(
            f"<section><h2>{html.escape(title)}</h2><table><thead><tr><th>#</th>"
            f"<th>Component</th><th>System</th>"
            f"<th>Preference</th><th>BT strength</th><th>vs avg.</th><th>N</th></tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table><p>Repeat agreement: {rendered_rate}</p></section>"
        )
    if result["strata"]:
        lines.extend(["## Stratified results", ""])
        cards.append("<h1>Stratified results</h1>")
    for dimension, values in result["strata"].items():
        for value, stratum in values.items():
            heading = f"{dimension} = {value}"
            lines.extend([f"### {heading}", "", f"Trials: **{stratum['trials']}**", ""])
            sections = []
            for criterion, rows in stratum["rankings"].items():
                criterion_title = criterion.replace("_", " ").title()
                lines.extend(
                    [
                        f"#### {criterion_title}",
                        "",
                        "| Rank | System | Preference rate | N |",
                        "| ---: | --- | ---: | ---: |",
                    ]
                )
                body = []
                for row in rows:
                    rank = row["rank"] if row["rank"] is not None else "—"
                    lines.append(
                        f"| {rank} | {row['system']} | {row['preference_rate']:.1%} | "
                        f"{row['comparisons']} |"
                    )
                    body.append(
                        f"<tr><td>{rank}</td><td>{html.escape(str(row['system']))}</td>"
                        f"<td>{row['preference_rate']:.1%}</td><td>{row['comparisons']}</td></tr>"
                    )
                lines.append("")
                sections.append(
                    f"<h3>{html.escape(criterion_title)}</h3><table><thead><tr><th>#</th>"
                    f"<th>System</th><th>Preference</th><th>N</th></tr></thead>"
                    f"<tbody>{''.join(body)}</tbody></table>"
                )
            cards.append(
                f"<section><h2>{html.escape(heading)}</h2><p>{stratum['trials']} trials</p>"
                f"{''.join(sections)}</section>"
            )
    (output_dir / "report.md").write_text("\n".join(lines))
    document = f"""<!doctype html><html><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width\"><title>Listening results</title><style>
body{{font-family:ui-monospace,monospace;background:#f2eddf;color:#191812;margin:0;padding:48px}}main{{max-width:1000px;margin:auto}}h1,h2{{font-family:Georgia,serif}}h1{{font-size:54px}}section{{background:#fbf8ef;border:2px solid #191812;box-shadow:7px 7px 0 #191812;padding:24px;margin:28px 0}}table{{border-collapse:collapse;width:100%}}th,td{{padding:10px;border-bottom:1px solid #aaa;text-align:left}}th{{font-size:11px;text-transform:uppercase}}</style></head><body><main><h1>{html.escape(str(result['title']))}</h1><p>{result['raters']} raters · {result['trials_per_rater']} trials each</p>{''.join(cards)}</main></body></html>"""
    (output_dir / "report.html").write_text(document)
