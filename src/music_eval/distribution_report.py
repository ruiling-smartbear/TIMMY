# ruff: noqa: E501
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def _row(system: str, metrics: dict[str, Any]) -> str:
    manifold = metrics["manifold"] or {}
    return (
        f"| {system} | {metrics['candidate_samples']} | "
        f"{metrics['frechet_embedding_distance']:.4f} | "
        f"{metrics['kernel_audio_distance']:.4f} | "
        f"{metrics['candidate_diversity_mean_cosine_distance']:.4f} | "
        f"{metrics['candidate_to_reference_nearest']['mean']:.4f} | "
        f"{metrics['reference_to_candidate_nearest']['mean']:.4f} | "
        f"{manifold.get('precision', '—')} | {manifold.get('recall', '—')} |"
    )


def write_distribution_report(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    lines = [
        "# Music distribution report",
        "",
        f"Embedding: **{result['embedding_model']}** · checkpoint: `{result['checkpoint']}` · "
        f"dimension: **{result['embedding_dimension']}**",
        "",
        f"Reference system: **{result['reference_system']}**",
        "",
        "> Fréchet embedding distance is not automatically FAD. Its interpretation depends on the "
        "declared embedding model, checkpoint, preprocessing, and corpus protocol.",
        "",
        "| System | N | Fréchet ↓ | KAD ↓ | Diversity | Candidate→ref ↓ | Ref→candidate ↓ | Precision | Recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for system, metrics in result["comparisons"].items():
        lines.append(_row(system, metrics))
    for dimension, values in result["strata"].items():
        lines.extend(["", f"## By {dimension}", ""])
        for value, comparisons in values.items():
            lines.extend(
                [
                    f"### {value}",
                    "",
                    "| System | N | Fréchet ↓ | KAD ↓ | Diversity | Candidate→ref ↓ | Ref→candidate ↓ | Precision | Recall |",
                    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for system, metrics in comparisons.items():
                lines.append(_row(system, metrics))
            lines.append("")
    (output_dir / "report.md").write_text("\n".join(lines))

    def table(comparisons: dict[str, Any]) -> str:
        rows = []
        for system, metrics in comparisons.items():
            manifold = metrics["manifold"] or {}
            rows.append(
                f"<tr><td>{html.escape(system)}</td><td>{metrics['candidate_samples']}</td>"
                f"<td>{metrics['frechet_embedding_distance']:.4f}</td>"
                f"<td>{metrics['kernel_audio_distance']:.4f}</td>"
                f"<td>{metrics['candidate_diversity_mean_cosine_distance']:.4f}</td>"
                f"<td>{metrics['candidate_to_reference_nearest']['mean']:.4f}</td>"
                f"<td>{metrics['reference_to_candidate_nearest']['mean']:.4f}</td>"
                f"<td>{manifold.get('precision', '—')}</td><td>{manifold.get('recall', '—')}</td></tr>"
            )
        return "<table><thead><tr><th>System</th><th>N</th><th>Fréchet ↓</th><th>KAD ↓</th><th>Diversity</th><th>Candidate→ref ↓</th><th>Ref→candidate ↓</th><th>Precision</th><th>Recall</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"

    sections = [f"<section><h2>All samples</h2>{table(result['comparisons'])}</section>"]
    for dimension, values in result["strata"].items():
        for value, comparisons in values.items():
            sections.append(
                f"<section><p class='eyebrow'>{html.escape(dimension)}</p>"
                f"<h2>{html.escape(value)}</h2>{table(comparisons)}</section>"
            )
    document = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Music distribution report</title><style>body{{margin:0;background:#11140f;color:#edf5dd;font-family:Menlo,monospace;padding:52px}}main{{max-width:1180px;margin:auto}}h1,h2{{font-family:"Iowan Old Style",Baskerville,serif}}h1{{font-size:58px;margin-bottom:8px}}.lede,.eyebrow{{color:#a8b49b}}.eyebrow{{text-transform:uppercase;letter-spacing:.15em;font-size:11px}}section{{margin:36px 0;padding:24px;border:1px solid #59634f;background:#181d15;overflow:auto}}table{{width:100%;border-collapse:collapse;white-space:nowrap}}th,td{{padding:11px;text-align:left;border-bottom:1px solid #394033}}th{{color:#cce976;font-size:11px;text-transform:uppercase}}</style></head><body><main><p class="eyebrow">music-eval / corpus evidence</p><h1>Distribution report</h1><p class="lede">{html.escape(result['embedding_model'])} · {html.escape(result['checkpoint'])} · reference: {html.escape(result['reference_system'])}</p>{''.join(sections)}</main></body></html>"""
    (output_dir / "report.html").write_text(document)
