#!/usr/bin/env python3
from __future__ import annotations

import html
import json

from v8_common import ROOT
from v8_frozen import verify_frozen


OUTPUT = ROOT / "artifacts/reports"


def _fmt(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def run():
    manifest = verify_frozen()
    dataset = json.loads((ROOT / "artifacts/studies/v8_safety_dataset/dataset_summary.json").read_text())
    validation = json.loads((ROOT / "artifacts/studies/v8_policy_validation/summary.json").read_text())
    state_action = json.loads((ROOT / "artifacts/studies/v8_safety_model_selection/state_vs_action.json").read_text())
    final = json.loads((ROOT / "artifacts/studies/v8_micro_holdout/summary.json").read_text())
    osm = json.loads((ROOT / "artifacts/studies/v8_real_topology/summary.json").read_text())
    json.loads((ROOT / "artifacts/studies/v8_failure_analysis/summary.json").read_text())
    v8 = final["comparison"]["V8-F"]
    rows = [
        ("Development paired conditions", dataset["pair_count"]),
        ("Unsafe development conditions", dataset["unsafe_intervention_count"]),
        ("Final microscopic pairs", final["pair_count"]),
        ("Safety model PR-AUC", final["safety_classifier"]["pr_auc_average_precision"]),
        ("Safety model unsafe recall", final["safety_classifier"]["unsafe_recall"]),
        ("V8-F interventions", v8["intervention_count"]),
        ("V8-F precision", v8["deployment_precision"]),
        ("V8-F coverage", v8["coverage"]),
        ("V8-F ORR", v8["opportunity_realization_rate"]),
        ("V8-F safety violations", v8["safety_violation_count"]),
        ("Safe-success retention", final["safety_cost_of_uplift"]["safe_success_retention"]),
        ("OSM paired conditions", osm["paired_condition_count"]),
        ("OSM V8-F interventions", osm["primary_metrics"]["intervention_count"]),
        ("OSM V8-F safe successes", osm["primary_metrics"]["success_count"]),
    ]
    markdown = [
        "# CONCORDIA v8 — Safety-Filtered Uplift Ranking",
        "",
        "## Registered design",
        "",
        "The immutable v7 direct paired random-forest traffic ranker is followed by a calibrated, pre-decision action-aware unsafe-intervention veto. The unsafe event is `Risk(Adaptive) > Risk(B1) + 0.25`; predicted regret above 0.08 and illegal actions are also vetoed. No RL was used.",
        "",
        "## Evidence summary",
        "",
        "| Measure | Result |",
        "|---|---:|",
        *[f"| {name} | {_fmt(value)} |" for name, value in rows],
        "",
        "## Validation selection",
        "",
        f"The registered lexicographic search evaluated {validation['candidate_count']} policy points and found {validation['feasible_candidate_count']} feasible points. Safe abstention was `{validation['safe_abstention']}`.",
        "",
        "## State vs action-aware safety",
        "",
        f"State-only validation PR-AUC was {_fmt(state_action['state_only_pr_auc'])}; the selected action-aware model achieved {_fmt(state_action['action_aware_pr_auc'])}. This comparison is reported as measured, without promoting a failed hypothesis.",
        "",
        "## Traffic ranking",
        "",
        f"On the untouched final set, the population mean paired traffic uplift was {_fmt(final['traffic_ranking']['population_mean_uplift'])}; the top 10% mean was {_fmt(final['traffic_ranking']['top_k']['top_10_percent']['mean_realized_uplift'])}, with Spearman correlation {_fmt(final['traffic_ranking']['spearman'])}.",
        "",
        "## Transfer and limits",
        "",
        f"The OSM bridge used {osm['od_pair_count']} newly selected Gangnam OD pairs, {osm['paired_condition_count']} paired conditions, and synthetic demand. It is evidence on real geometry, not observed Seoul traffic. Failure taxonomy and mechanism-proxy plots are in `artifacts/studies/v8_failure_analysis/`.",
        "",
        "## Reproducibility",
        "",
        f"Freeze manifest self-hash: `{manifest['manifest_self_hash']}`. The final microscopic and OSM evaluations verified the same frozen hash before and after evaluation.",
    ]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    md_path = OUTPUT / "CONCORDIA_v8_final_report.md"
    md_path.write_text("\n".join(markdown) + "\n")
    table = "".join(f"<tr><td>{html.escape(name)}</td><td>{html.escape(_fmt(value))}</td></tr>" for name, value in rows)
    html_path = OUTPUT / "CONCORDIA_v8_final_report.html"
    html_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>CONCORDIA v8</title>"
        "<style>body{font-family:system-ui;background:#07111f;color:#eef6ff;max-width:1000px;margin:40px auto;padding:0 24px}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #29415c}h1,h2{color:#62ddff}.note{color:#a9bed3}</style></head><body>"
        "<h1>CONCORDIA v8 — Safety-Filtered Uplift Ranking</h1>"
        "<p>Frozen v7 traffic ranking followed by calibrated action-aware unsafe veto. No RL.</p>"
        f"<table><tr><th>Measure</th><th>Result</th></tr>{table}</table>"
        f"<h2>Scope</h2><p class='note'>{html.escape(osm['claim_boundary'])}</p>"
        f"<p>Freeze manifest: <code>{manifest['manifest_self_hash']}</code></p></body></html>"
    )
    print(md_path)


if __name__ == "__main__":
    run()
