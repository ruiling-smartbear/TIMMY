# ruff: noqa: E501
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from music_eval.text import markdown_cell

_SEPARATED_NOTE = (
    "* Separated data: the Bradley-Terry maximum-likelihood estimate does not exist "
    "(some system or group of systems never lost or never won), so log strengths are the "
    "ridge-regularized estimate and should be read as an ordering only."
)


def _rank_cell(row: dict[str, Any]) -> str:
    rank = "—" if row["rank"] is None else str(row["rank"])
    return f"{rank}*" if row["separated"] else rank


def write_listening_report(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
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
    bootstrap = result["bootstrap"]
    if bootstrap["enabled"]:
        lines.extend(
            [
                f"95% intervals: **{bootstrap['samples']} rater-level bootstrap samples** "
                f"(seed `{bootstrap['seed']}`). They capture rater sampling uncertainty, "
                "not prompt/corpus sampling uncertainty.",
                "",
            ]
        )
    else:
        lines.extend([f"> Confidence intervals unavailable: {bootstrap['reason']}.", ""])
    cards = []
    for criterion, rows in result["rankings"].items():
        title = criterion.replace("_", " ").title()
        lines.extend(
            [
                f"## {title}",
                "",
                "| Rank | Component | System | Preference rate (95% CI) | BT log strength (95% CI) | vs average | N |",
                "| ---: | ---: | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        body = []
        for row in rows:
            rank = _rank_cell(row)
            preference_ci = row.get("preference_rate_ci95")
            strength_ci = row.get("log_strength_ci95")
            rendered_preference_ci = (
                "—"
                if preference_ci is None
                else f"[{preference_ci[0]:.1%}, {preference_ci[1]:.1%}]"
            )
            rendered_strength_ci = (
                "—"
                if strength_ci is None
                else f"[{strength_ci[0]:.3f}, {strength_ci[1]:.3f}]"
            )
            lines.append(
                f"| {rank} | {row['component']} | {markdown_cell(row['system'])} | "
                f"{row['preference_rate']:.1%} {rendered_preference_ci} | "
                f"{row['log_strength']:.3f} {rendered_strength_ci} | "
                f"{row['vs_average']:.1%} | {row['comparisons']} |"
            )
            body.append(
                f"<tr><td>{rank}</td><td>{row['component']}</td>"
                f"<td>{html.escape(str(row['system']))}</td>"
                f"<td>{row['preference_rate']:.1%}<small>{rendered_preference_ci}</small></td>"
                f"<td>{row['log_strength']:.3f}<small>{rendered_strength_ci}</small></td>"
                f"<td>{row['vs_average']:.1%}</td><td>{row['comparisons']}</td></tr>"
            )
        separated_note = ""
        if any(row["separated"] for row in rows):
            lines.extend(["", "\\" + _SEPARATED_NOTE])
            separated_note = f"<p>{html.escape(_SEPARATED_NOTE)}</p>"
        reliability = result["repeat_reliability"][criterion]
        rate = reliability["agreement_rate"]
        rendered_rate = "not measured" if rate is None else f"{rate:.1%}"
        inter_rater = result["inter_rater_agreement"][criterion]
        inter_rate = inter_rater["pairwise_agreement_rate"]
        rendered_inter = "not measured" if inter_rate is None else f"{inter_rate:.1%}"
        side = result["side_choice_diagnostics"][criterion]
        side_rate = side["side_a_rate_among_non_ties"]
        rendered_side = "not measured" if side_rate is None else f"{side_rate:.1%}"
        lines.extend(
            [
                "",
                f"Repeat-trial agreement: **{rendered_rate}** · inter-rater pair agreement: "
                f"**{rendered_inter}** · raw side-A selection among non-ties: "
                f"**{rendered_side}**",
                "",
            ]
        )
        cards.append(
            f"<section><h2>{html.escape(title)}</h2><table><thead><tr><th>#</th>"
            f"<th>Component</th><th>System</th>"
            f"<th>Preference</th><th>BT strength</th><th>vs avg.</th><th>N</th></tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table>{separated_note}<p>Repeat agreement: {rendered_rate} · "
            f"inter-rater agreement: {rendered_inter} · side-A rate: {rendered_side}</p></section>"
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
                    rank = _rank_cell(row)
                    lines.append(
                        f"| {rank} | {markdown_cell(row['system'])} | {row['preference_rate']:.1%} | "
                        f"{row['comparisons']} |"
                    )
                    body.append(
                        f"<tr><td>{rank}</td><td>{html.escape(str(row['system']))}</td>"
                        f"<td>{row['preference_rate']:.1%}</td><td>{row['comparisons']}</td></tr>"
                    )
                separated_note = ""
                if any(row["separated"] for row in rows):
                    lines.extend(["", "\\" + _SEPARATED_NOTE])
                    separated_note = f"<p>{html.escape(_SEPARATED_NOTE)}</p>"
                lines.append("")
                sections.append(
                    f"<h3>{html.escape(criterion_title)}</h3><table><thead><tr><th>#</th>"
                    f"<th>System</th><th>Preference</th><th>N</th></tr></thead>"
                    f"<tbody>{''.join(body)}</tbody></table>{separated_note}"
                )
            cards.append(
                f"<section><h2>{html.escape(heading)}</h2><p>{stratum['trials']} trials</p>"
                f"{''.join(sections)}</section>"
            )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    document = f"""<!doctype html><html><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width\"><title>Listening results</title><style>
body{{font-family:ui-monospace,monospace;background:#f2eddf;color:#191812;margin:0;padding:48px}}main{{max-width:1100px;margin:auto}}h1,h2{{font-family:Georgia,serif}}h1{{font-size:54px}}section{{background:#fbf8ef;border:2px solid #191812;box-shadow:7px 7px 0 #191812;padding:24px;margin:28px 0;overflow:auto}}table{{border-collapse:collapse;width:100%}}th,td{{padding:10px;border-bottom:1px solid #aaa;text-align:left}}th{{font-size:11px;text-transform:uppercase}}small{{display:block;color:#716d61;margin-top:4px;white-space:nowrap}}</style></head><body><main><h1>{html.escape(str(result['title']))}</h1><p>{result['raters']} raters · {result['trials_per_rater']} trials each</p>{''.join(cards)}</main></body></html>"""
    (output_dir / "report.html").write_text(document, encoding="utf-8")
