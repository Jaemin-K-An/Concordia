#!/usr/bin/env python3
from __future__ import annotations

import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/reports/final_report_v4.html"
CANONICAL = ROOT / "artifacts/report.html"


def _load(relative: str):
    path = ROOT / relative
    if not path.is_file():
        raise RuntimeError(f"required v4 evidence is missing: {relative}")
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _figure(relative: str, caption: str) -> str:
    path = ROOT / relative
    if not path.is_file():
        return ""
    source = Path("..") / Path(relative).relative_to("artifacts")
    return (
        f'<figure><img src="{html.escape(str(source))}" alt="{html.escape(caption)}">'
        f"<figcaption>{html.escape(caption)}</figcaption></figure>"
    )


def run() -> Path:
    model = _load("artifacts/studies/v4_model_selection/summary.json")
    calibration = _load(
        "artifacts/studies/v4_precision_validation/calibration_summary.json"
    )
    benefit = _load("artifacts/studies/v4_precision_validation/benefit_summary.json")
    safety = _load("artifacts/studies/v4_precision_validation/safety_summary.json")
    validation = _load("artifacts/studies/v4_precision_validation/summary.json")
    validation_tests = _load(
        "artifacts/studies/v4_precision_validation/statistical_tests.json"
    )
    holdout = _load("artifacts/studies/v4_frozen_holdout/summary.json")
    microscopic = _load("artifacts/studies/v4_microscopic/summary.json")
    real = _load("artifacts/studies/v4_real_topology/summary.json")
    stress = _load("artifacts/studies/v4_stress/summary.json")
    primary = holdout["primary_metrics"]

    policy_rows = "".join(
        "<tr>"
        f"<td>{html.escape(name)}</td>"
        f"<td>{metrics['intervention_count']}</td>"
        f"<td>{_pct(metrics['intervention_precision'])}</td>"
        f"<td>{_pct(metrics['coverage'])}</td>"
        f"<td>{metrics['mean_network_ttt_gain']:.5f}</td>"
        "</tr>"
        for name, metrics in holdout["policy_metrics"].items()
    )
    group_rows = []
    for dimension, groups in holdout["group_metrics"]["dimensions"].items():
        for group, metrics in groups.items():
            precision = (
                "—" if metrics["precision"] is None else _pct(metrics["precision"])
            )
            group_rows.append(
                "<tr>"
                f"<td>{html.escape(dimension)}</td>"
                f"<td>{html.escape(group)}</td>"
                f"<td>{metrics['intervention_count']}</td>"
                f"<td>{precision}</td>"
                f"<td>{_pct(metrics['coverage'])}</td>"
                "</tr>"
            )
    overlap_rows = "".join(
        "<tr>"
        f"<td>{html.escape(group)}</td>"
        f"<td>{metrics['intervention_count']}</td>"
        f"<td>{_pct(metrics['intervention_precision'])}</td>"
        f"<td>{_pct(metrics['coverage'])}</td>"
        "</tr>"
        for group, metrics in real["activation_by_overlap_class"].items()
    )
    figures = "".join(
        _figure(path, caption)
        for path, caption in (
            (
                "artifacts/studies/v4_precision_validation/figures/calibration_comparison.png",
                "Calibration comparison on development calibration data",
            ),
            (
                "artifacts/studies/v4_precision_validation/figures/precision_at_coverage.png",
                "Precision–coverage frontier before freeze",
            ),
            (
                "artifacts/studies/v4_frozen_holdout/figures/holdout_precision_coverage.png",
                "Untouched holdout policy comparison",
            ),
            (
                "artifacts/studies/v4_microscopic/figures/microscopic_v4_selectivity.png",
                "Actual SUMO microscopic selectivity",
            ),
            (
                "artifacts/studies/v4_real_topology/figures/activation_by_overlap.png",
                "Activation across three real-geometry OD overlap classes",
            ),
            (
                "artifacts/studies/v4_stress/figures/stress_tail.png",
                "Frozen-policy distribution-shift tail",
            ),
        )
    )
    h19 = validation_tests["H19_interaction"]
    h20 = validation_tests
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>CONCORDIA v4 final report</title>
<style>
:root{{--ink:#121915;--muted:#627067;--line:#d7dfd9;--paper:#f8fbf8;--accent:#176149;--pale:#e9f4ee}}
body{{font:16px/1.58 Inter,-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;max-width:1160px;margin:auto;padding:48px 28px;color:var(--ink);background:var(--paper)}}
h1{{font-size:48px;line-height:1;letter-spacing:-.045em;margin:8px 0}}h2{{margin-top:46px;letter-spacing:-.025em}}.eyebrow{{font-size:12px;letter-spacing:.15em;text-transform:uppercase;color:var(--accent);font-weight:800}}.lead{{font-size:21px;max-width:900px}}.boundary{{border-left:5px solid var(--accent);padding:15px 18px;background:var(--pale)}}
.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}}.metric{{background:white;border:1px solid var(--line);padding:16px}}.metric b{{display:block;font-size:28px}}.metric span{{font-size:13px;color:var(--muted)}}
table{{border-collapse:collapse;width:100%;margin:18px 0}}th,td{{border-bottom:1px solid var(--line);padding:9px;text-align:left;vertical-align:top}}th{{font-size:12px;text-transform:uppercase;letter-spacing:.06em}}.decision{{border:1px solid var(--ink);padding:24px;font-size:23px;font-weight:750}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px}}figure{{margin:0;padding:10px;background:white;border:1px solid var(--line)}}img{{width:100%}}figcaption{{font-size:13px;color:var(--muted)}}code{{background:#e9eeea;padding:2px 5px}}
</style></head><body>
<div class="eyebrow">Precision-constrained coverage optimization</div>
<h1>CONCORDIA v4</h1>
<p class="lead">Estimate intervention success, traffic benefit, and safety risk separately; calibrate under distribution shift; then maximize coverage subject to a preregistered precision constraint.</p>
<p class="boundary"><strong>Evidence boundary.</strong> {html.escape(holdout['claim_boundary'])} The new {holdout['case_count']}-case analytical holdout was never used for fitting or threshold selection. SUMO studies use actual microscopic simulation with synthetic demand. Real-topology results use committed OSM geometry with synthetic OD demand. RL is not used.</p>

<h2>1. Frozen v4 decision</h2>
<div class="metrics">
<div class="metric"><b>{_pct(primary['intervention_precision'])}</b><span>Holdout intervention precision</span></div>
<div class="metric"><b>{_pct(primary['coverage'])}</b><span>Holdout coverage</span></div>
<div class="metric"><b>{primary['intervention_count']}</b><span>Interventions</span></div>
<div class="metric"><b>{primary['mean_network_ttt_gain']:.5f}</b><span>Mean network TTT gain</span></div>
<div class="metric"><b>{_pct(primary['population_benefit_rate'])}</b><span>Population benefit rate</span></div>
<div class="metric"><b>{_pct(holdout['group_metrics']['worst_group_precision'])}</b><span>Worst activated-group precision</span></div>
</div>
<p>Precision 95% Wilson interval: [{_pct(primary['intervention_precision_ci95'][0])}, {_pct(primary['intervention_precision_ci95'][1])}]. Safety, regret, and legal violation counts among interventions were {primary['safety_violation_count']}, {primary['regret_violation_count']}, and {primary['legal_violation_count']}.</p>

<h2>2. Model selection and calibration</h2>
<p>Robust CV selected <strong>{html.escape(model['selected_model'])}</strong> using worst-group and lower-tail precision before coverage, ECE, and mean selected gain. CV worst-group precision was {_pct(model['worst_group_precision'])}; mean coverage was {_pct(model['mean_coverage'])}. The probability calibrator was <strong>{html.escape(calibration['selected_method'])}</strong>, with ECE {calibration['selected_ece']:.4f} against the preregistered 0.05 target. Benefit models were <code>{html.escape(benefit['selected_mean_model'])}</code> and <code>{html.escape(benefit['selected_lower_model'])}</code>. The calibration false-safe count for the conservative safety UCB was {safety['calibration_false_safe_metrics']['false_safe_count']}.</p>
<p>The frozen policy was <strong>{html.escape(validation['selected_policy'])}</strong>. Validation-only H20 Coverage@Precision80 was {_pct(h20['H20_ESIV_coverage_at_precision80'])} for ESIV and {_pct(h20['H20_probability_coverage_at_precision80'])} for probability gating. H19 interaction coefficient/effect: {html.escape(str(h19))}.</p>

<h2>3. Untouched holdout comparisons</h2>
<table><tr><th>Policy</th><th>Interventions</th><th>Precision</th><th>Coverage</th><th>Mean TTT gain</th></tr>{policy_rows}</table>

<h2>4. Worst-group audit</h2>
<p>Median activated-group precision was {_pct(holdout['group_metrics']['median_group_precision'])}; worst activated-group precision was {_pct(holdout['group_metrics']['worst_group_precision'])}. A dash means the frozen policy made no intervention in that group.</p>
<table><tr><th>Dimension</th><th>Group</th><th>N intervention</th><th>Precision</th><th>Coverage</th></tr>{''.join(group_rows)}</table>

<h2>5. Actual microscopic and real-geometry validation</h2>
<p>The actual SUMO matrix contained {microscopic['pair_count']} paired cases. V4-F made {microscopic['policy_metrics']['V4-F']['intervention_count']} interventions with precision {_pct(microscopic['policy_metrics']['V4-F']['intervention_precision'])}; adaptive success is claimed only when that count is positive. Phantom-jam prediction remained secondary and was not a v4 gate.</p>
<p>The real-geometry study used {len(real['od_pairs'])} passenger-legal OD pairs spanning low, medium, and high route overlap. Demand is synthetic and the result is mechanism-transfer evidence, not a Seoul effect estimate.</p>
<table><tr><th>Overlap class</th><th>Interventions</th><th>Precision</th><th>Coverage</th></tr>{overlap_rows}</table>

<h2>6. Distribution-shift stress</h2>
<p>Across {stress['case_count']} post-holdout stress cases, precision was {_pct(stress['policy_metrics']['intervention_precision'])}, coverage {_pct(stress['policy_metrics']['coverage'])}, safety violations {stress['policy_metrics']['safety_violation_count']}, and loss CVaR {stress['statistics']['loss_cvar']:.5f}. Model and thresholds remained immutable.</p>

<h2>7. Figures</h2><div class="grid">{figures}</div>

<h2>8. Final decision</h2>
<p class="decision">Outcome {html.escape(holdout['outcome'])} — {html.escape(holdout['outcome_text'])}<br>Always-on adaptive navigation remains rejected as a universal policy.</p>
</body></html>"""
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(document, encoding="utf-8")
    CANONICAL.write_text(
        document.replace('src="../studies/', 'src="studies/'),
        encoding="utf-8",
    )
    print(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    run()
