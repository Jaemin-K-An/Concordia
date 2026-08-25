from __future__ import annotations

import html
import json
from pathlib import Path
from typing import List

from concordia.errors import ValidationError


def build_report(runs_root: str, output: str) -> Path:
    root = Path(runs_root)
    records: List[dict] = []
    for metrics_path in sorted(root.glob("*/metrics.json")):
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        records.append({"run_id": metrics_path.parent.name, **metrics})
    if not records:
        raise ValidationError(f"no experiment runs found under {root}")
    rows = []
    for record in records:
        rows.append(
            "<tr>"
            f"<td>{html.escape(record['run_id'])}</td>"
            f"<td>{html.escape(str(record['scenario']))}</td>"
            f"<td>{record['demand']:.3f}</td>"
            f"<td>{record['private_best_ttt']['mean']:.3f}</td>"
            f"<td>{record['proposed_ttt']['mean']:.3f}</td>"
            f"<td>{record['price_of_anarchy']:.5f}</td>"
            f"<td>{record['proposed_regret']['p95']:.6f}</td>"
            "</tr>"
        )
    document = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Concordia experiment report</title>
<style>body{font:16px system-ui;max-width:1100px;margin:3rem auto;padding:0 1rem;color:#17202a}
table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccd1d1;padding:.55rem;text-align:right}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}.note{background:#f4f6f7;padding:1rem}</style>
</head><body><h1>Concordia experiment report</h1>
<p class="note"><strong>Scope:</strong> Synthetic analytical results. Safety values are route-level
surrogate exposure, not collision probability. Statistical claims require a pre-registered full matrix.</p>
<table><thead><tr><th>Run</th><th>Scenario</th><th>Demand</th><th>Preference-only TTT</th>
<th>Aligned TTT</th><th>PoA</th><th>Regret p95</th></tr></thead><tbody>
""" + "\n".join(rows) + "</tbody></table></body></html>\n"
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document, encoding="utf-8")
    return destination
