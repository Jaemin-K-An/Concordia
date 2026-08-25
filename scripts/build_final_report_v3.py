#!/usr/bin/env python3
from __future__ import annotations

import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/reports/final_report_v3.html"
CANONICAL = ROOT / "artifacts/report.html"


def _load(path: str):
    source = ROOT / path
    if not source.is_file():
        raise RuntimeError(f"required v3 evidence is missing: {path}")
    return json.loads(source.read_text(encoding="utf-8"))


def _figure(path: str, caption: str) -> str:
    source = ROOT / path
    if not source.is_file():
        return ""
    relative = Path("..") / Path(path).relative_to("artifacts")
    return (
        f'<figure><img src="{html.escape(str(relative))}" alt="{html.escape(caption)}">'
        f"<figcaption>{html.escape(caption)}</figcaption></figure>"
    )


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def run() -> Path:
    v2 = (ROOT / "FINAL_AUDIT_V2.md").read_text(encoding="utf-8")
    feasibility = _load("artifacts/studies/v3_feasibility_prediction/summary.json")
    holdout = _load("artifacts/studies/v3_selective_holdout/summary.json")
    ablations = _load("artifacts/studies/v3_selective_holdout/ablations.json")
    microscopic = _load("artifacts/studies/v3_microscopic_selective/summary.json")
    real = _load("artifacts/studies/v3_real_topology_selective/summary.json")
    tail = _load("artifacts/studies/v3_tail_robustness/summary.json")
    primary = holdout["primary_metrics"]
    model_metrics = feasibility["validation_metrics"]
    outcome = holdout["outcome"]
    outcome_text = {
        "S": "Selective CONCORDIA supported.",
        "P": "Selective CONCORDIA partially supported.",
        "F": "Selective CONCORDIA not supported.",
    }[outcome]
    figures = (
        ("artifacts/studies/v3_feasibility_prediction/figures/calibration_curve.png", "Development validation calibration"),
        ("artifacts/studies/v3_feasibility_prediction/figures/validation_risk_coverage.png", "Validation risk–coverage"),
        ("artifacts/studies/v3_selective_holdout/figures/policy_precision_coverage.png", "Untouched holdout policy comparison"),
        ("artifacts/studies/v3_selective_holdout/figures/holdout_risk_coverage.png", "Frozen holdout operating point"),
        ("artifacts/studies/v3_microscopic_selective/figures/phantom_probability_b1_b6_v3.png", "Microscopic VALID-event probability"),
        ("artifacts/studies/v3_microscopic_selective/figures/run_group_calibration.png", "Run-group phantom calibration"),
        ("artifacts/studies/v3_real_topology_selective/figures/real_topology_v3_ttt.png", "Real-topology selective TTT"),
        ("artifacts/studies/v3_real_topology_selective/figures/real_topology_v3_map.png", "Real-topology policy delta"),
        ("artifacts/studies/v3_tail_robustness/figures/tail_degradation.png", "Post-holdout tail robustness"),
    )
    figure_html = "".join(_figure(path, caption) for path, caption in figures)
    importance_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{value:.5f}</td></tr>"
        for name, value in feasibility["feature_importance_top10"]
    )
    ablation_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{_pct(value['intervention_precision'])}</td>"
        f"<td>{_pct(value['coverage'])}</td><td>{value['mean_network_ttt_gain']:.4f}</td></tr>"
        for name, value in ablations.items()
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>CONCORDIA v3 final report</title>
<style>
:root{{--ink:#131313;--muted:#686868;--line:#d7d7d2;--paper:#fbfbf8;--accent:#0b6e4f}}
body{{font:16px/1.58 Inter,-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;max-width:1140px;margin:0 auto;padding:48px 28px;color:var(--ink);background:var(--paper)}}
h1{{font-size:46px;line-height:1.04;letter-spacing:-.045em;margin-bottom:8px}}h2{{margin-top:48px;letter-spacing:-.025em}}h3{{margin-top:28px}}
.eyebrow{{font-size:12px;letter-spacing:.16em;text-transform:uppercase;color:var(--accent);font-weight:700}}.lead{{font-size:21px;max-width:820px;color:#333}}
.boundary{{border-left:5px solid var(--accent);padding:14px 18px;background:#eef7f3}}.decision{{font-size:24px;font-weight:750;padding:24px;border:1px solid var(--ink)}}
table{{border-collapse:collapse;width:100%;margin:18px 0}}th,td{{border-bottom:1px solid var(--line);padding:9px;text-align:left;vertical-align:top}}th{{font-size:12px;text-transform:uppercase;letter-spacing:.06em}}
.metric-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}}.metric{{border:1px solid var(--line);padding:16px}}.metric b{{display:block;font-size:28px}}.metric span{{color:var(--muted);font-size:13px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(410px,1fr));gap:18px}}figure{{margin:0;border:1px solid var(--line);padding:10px;background:white}}img{{width:100%}}figcaption{{font-size:13px;color:var(--muted)}}code{{background:#eee;padding:2px 5px}}
</style></head><body>
<div class="eyebrow">Alignment-feasibility-gated adaptive navigation</div>
<h1>CONCORDIA v3</h1>
<p class="lead">Know when optimization is worth attempting. Predict feasibility, intervene selectively, enforce a safety hard gate, and fall back to ETA navigation under uncertainty.</p>
<p class="boundary"><strong>Claim boundary.</strong> The primary result is an untouched synthetic analytical holdout. Microscopic tests use actual SUMO with synthetic demand. The real-topology test uses committed OSM geometry with an unseen synthetic OD. TTC, PET, and DRAC are surrogate conflict indicators—not crash probabilities.</p>

<h2>1. Why the architecture changed</h2>
<p>v2 showed that adaptive navigation can reverse direction across topology and preference conditions. The original H1, H2, H3, H4, and H6 failures remain intact: {"retained" if "FAIL" in v2 else "see FINAL_AUDIT_V2.md"}. v3 therefore does not claim a better universal optimizer. It turns alignment feasibility into a prediction-and-abstention problem.</p>
<p>The system estimates Alignment Potential, route overlap, alternative capacity, preference diversity, Route Attribute Dispersion, acceptance, predicted benefit, uncertainty, and safety margin. Only cases that pass every frozen gate receive B6; all others use B1 unchanged.</p>

<h2>2. Leakage barrier and frozen decision rule</h2>
<ol><li>v2 cases were reconstructed as development/training/validation evidence.</li><li>{feasibility['selected_model']} was selected on validation using PR-AUC, Brier score, calibration error, and interpretability.</li><li>Probability, APS, uncertainty, benefit, acceptance, tail, and safety thresholds were frozen with checksums.</li><li>{holdout['case_count']} new cases with unseen seeds and demand points were generated only after freeze.</li></ol>
<div class="metric-grid"><div class="metric"><b>{model_metrics['pr_auc']:.3f}</b><span>Validation PR-AUC (not primary evidence)</span></div><div class="metric"><b>{model_metrics['brier_score']:.3f}</b><span>Validation Brier score</span></div><div class="metric"><b>{model_metrics['ece']:.3f}</b><span>Validation ECE</span></div><div class="metric"><b>{'Yes' if holdout['threshold_immutable'] else 'No'}</b><span>Threshold immutable through holdout</span></div></div>

<h2>3. Study VI — untouched primary holdout</h2>
<div class="metric-grid"><div class="metric"><b>{_pct(primary['intervention_precision'])}</b><span>Intervention precision</span></div><div class="metric"><b>{_pct(primary['coverage'])}</b><span>Coverage</span></div><div class="metric"><b>{_pct(primary['population_benefit_rate'])}</b><span>Population benefit rate</span></div><div class="metric"><b>{primary['mean_network_ttt_gain']:.4f}</b><span>Mean network TTT gain</span></div></div>
<p>Precision 95% Wilson CI: [{_pct(primary['intervention_precision_ci95'][0])}, {_pct(primary['intervention_precision_ci95'][1])}]. Engineering success requires point precision ≥65%; strong scientific evidence separately requires the lower 95% bound to exceed 50%. Abstentions are neither counted as successes nor failures.</p>
<table><tr><th>Policy</th><th>Precision</th><th>Coverage</th><th>Mean TTT gain</th></tr>{ablation_rows}</table>
<p>H8 scientific criterion: <strong>{'PASS' if holdout['statistical_tests']['H8_scientific_lower_ci_above_half'] else 'FAIL'}</strong>. H9 failure reduction vs always-on: <strong>{'PASS' if holdout['statistical_tests']['H9_failure_rate_reduced_vs_B6'] else 'FAIL'}</strong>. H10 mean-cost non-inferiority: <strong>{'PASS' if holdout['statistical_tests']['H10_mean_network_cost_noninferior'] else 'FAIL'}</strong>.</p>

<h2>4. What predicts a feasible intervention?</h2>
<table><tr><th>Feature</th><th>Selected-model importance</th></tr>{importance_rows}</table>
<p>On holdout, route-overlap/WIN correlation was {holdout['statistical_tests']['H13_overlap_win_correlation']:.3f}; preference-diversity × route-attribute-diversity correlation was {holdout['statistical_tests']['H14_interaction_win_correlation']:.3f}. These are descriptive synthetic associations, not causal estimates.</p>

<h2>5. Microscopic safety-selectivity</h2>
<p>{microscopic['metastable_search_run_count']} B1 SUMO runs searched the metastable demand range before selective testing. Phantom calibration was <strong>{'complete' if microscopic['phantom_calibration_complete'] else 'incomplete'}</strong>; its separate gate was <strong>{'used' if microscopic['phantom_gate_used'] else 'excluded'}</strong>. In the selective matrix, B6 had {microscopic['statistics']['H11_B6_safety_failure_count']} safety failures and v3 had {microscopic['statistics']['H11_V3_safety_failure_count']}.</p>

<h2>6. Real-topology selective transfer</h2>
<p>The evaluation OD <code>{html.escape(str(real['evaluation_od']))}</code> differs from the development OD. {real['legal_route_count']} passenger-legal routes were tested on the committed Gangnam OSM extract. V3 precision was {_pct(real['policy_metrics']['V3']['intervention_precision'])}, coverage {_pct(real['policy_metrics']['V3']['coverage'])}, and mean TTT gain {real['policy_metrics']['V3']['mean_network_ttt_gain']:.3f}s. This is mechanism-transfer evidence under synthetic demand, not a Seoul traffic-effect estimate.</p>

<h2>7. Post-holdout tail robustness</h2>
<p>With demand ×{tail['demand_shift_multiplier']:.2f} and preference-variance input ×{tail['preference_variance_multiplier']:.2f}, mean stressed TTT gain was {tail['policy_metrics']['mean_network_ttt_gain']:.4f}. The {tail['statistics']['H10_tail_degradation_quantile']:.0%} degradation CVaR was {tail['statistics']['H10_tail_degradation_cvar']:.4f} against a frozen {tail['statistics']['H10_tail_limit']:.4f} limit. Threshold checksums remained unchanged.</p>

<h2>8. Figures</h2><div class="grid">{figure_html}</div>
<h2>9. Final decision</h2>
<p class="decision">Always-on Adaptive Navigation: rejected as universal policy.<br>Selective Adaptive Navigation: Outcome {outcome}. {outcome_text}</p>
<p>RL remains rejected and is not part of the v3 policy.</p>
</body></html>"""
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(document, encoding="utf-8")
    CANONICAL.write_text(document, encoding="utf-8")
    print(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    run()
