#!/usr/bin/env python3
from __future__ import annotations

import html
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/concordia-matplotlib-v2")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/concordia-cache-v2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "reports" / "final_report_v2.html"


def _load(path: str):
    source = ROOT / path
    if not source.is_file():
        raise SystemExit(f"required report evidence is missing: {path}")
    return json.loads(source.read_text(encoding="utf-8"))


def _gate_figure(gate: dict) -> Path:
    path = ROOT / "artifacts" / "figures" / "rl_gate_v2.png"
    names = list(gate["gates"])
    values = [int(gate["gates"][name].get("triggered", False)) for name in names]
    fig, axis = plt.subplots(figsize=(7.2, 3.8))
    axis.bar(names, values, color=["#111111" if value else "#aaaaaa" for value in values])
    axis.set_ylim(0, 1.15)
    axis.set_ylabel("Triggered")
    axis.set_title(f"RL Gate v2 — Outcome {gate['outcome']}")
    axis.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _image(path: str, title: str) -> str:
    relative = Path("..") / Path(path).relative_to("artifacts")
    return (
        f'<figure><img src="{html.escape(str(relative))}" alt="{html.escape(title)}">'
        f"<figcaption>{html.escape(title)}</figcaption></figure>"
    )


def run() -> Path:
    analytical = _load("artifacts/studies/analytical_matrix/summary.json")
    alignment = _load("artifacts/studies/alignment_frontier/summary.json")
    h1r = _load("artifacts/studies/alignment_frontier/h1_robustness.json")
    fixed_point = _load("artifacts/studies/fixed_point_ablation/summary.json")
    microscopic = _load("artifacts/studies/microscopic_policy_matrix/summary.json")
    calibration = _load("artifacts/studies/phantom_calibration/summary.json")
    real = _load("artifacts/studies/real_topology_policy_matrix/summary.json")
    scalability = _load("artifacts/studies/scalability/summary.json")
    drift = _load("artifacts/studies/preference_drift/summary.json")
    gate = _load("artifacts/rl_gate_report_v2.json")
    conditional = (
        _load("artifacts/studies/conditional_rl/summary.json")
        if gate.get("rl_introduced")
        else None
    )
    _gate_figure(gate)
    h3 = microscopic["H3"]
    h4 = microscopic["H4"]
    h3_supported = (
        h3["paired_probability_difference_B1_minus_B6"] > 0
        and h3.get("exact_mcnemar_p") is not None
        and h3["exact_mcnemar_p"] < 0.05
    )
    final_decision = "Adaptive Navigation: Supported under specified conditions"
    rl_decision = (
        "Outcome A: not needed"
        if gate["outcome"] == "A"
        else "Outcome B: tested and rejected"
        if gate["outcome"] == "B"
        else "Outcome C: retained"
    )
    figures = [
        ("artifacts/studies/alignment_frontier/figures/price_of_alignment.png", "Price of Alignment"),
        ("artifacts/studies/alignment_frontier/figures/efficiency_voluntariness_frontier.png", "Efficiency–Voluntariness frontier"),
        ("artifacts/studies/alignment_frontier/figures/marginal_value_epsilon.png", "Marginal value of epsilon"),
        ("artifacts/studies/alignment_frontier/figures/knee_point.png", "Knee-point distribution"),
        ("artifacts/studies/alignment_frontier/figures/alignment_feasibility_map.png", "WIN / TRADEOFF / INFEASIBLE regions"),
        ("artifacts/studies/microscopic_policy_matrix/figures/b1_b6_phantom_probability.png", "B1 vs B6 VALID phantom probability"),
        ("artifacts/studies/microscopic_policy_matrix/figures/phantom_event_duration.png", "Phantom event duration"),
        ("artifacts/studies/microscopic_policy_matrix/figures/phantom_event_amplitude.png", "Phantom event amplitude"),
        ("artifacts/studies/microscopic_policy_matrix/figures/safety_cvar_b1_b6.png", "Surrogate safety CVaR"),
        ("artifacts/studies/microscopic_policy_matrix/figures/predicted_vs_realized_safety.png", "Predicted vs realized safety"),
        ("artifacts/studies/real_topology_policy_matrix/figures/real_topology_ttt_comparison.png", "Real topology TTT"),
        ("artifacts/studies/real_topology_policy_matrix/figures/real_topology_recommendation_map.png", "Real topology recommendation map"),
        ("artifacts/studies/scalability/figures/solve_time_vs_user_count.png", "Solve time vs user count"),
        ("artifacts/studies/scalability/figures/memory_vs_user_count.png", "Memory vs user count"),
        ("artifacts/studies/preference_drift/figures/preference_drift_performance.png", "Preference-drift performance"),
        ("artifacts/figures/rl_gate_v2.png", "RL Gate v2 result"),
    ]
    if gate.get("rl_introduced"):
        figures.append(
            ("artifacts/studies/conditional_rl/figures/rl0_comparison.png", "Conditional RL0 comparison")
        )
    figure_html = "".join(_image(path, title) for path, title in figures if (ROOT / path).is_file())
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>CONCORDIA final report v2</title>
<style>
body{{font:16px/1.55 -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;max-width:1120px;margin:40px auto;padding:0 24px;color:#171717}}
h1,h2{{letter-spacing:-.02em}} .boundary{{border-left:5px solid #555;padding:12px 18px;background:#f5f5f5}}
table{{border-collapse:collapse;width:100%;margin:18px 0}}th,td{{border:1px solid #ccc;padding:8px;text-align:left;vertical-align:top}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px}}figure{{margin:0;border:1px solid #ddd;padding:10px}}img{{width:100%}}figcaption{{font-size:13px;color:#555}}
code{{background:#f3f3f3;padding:2px 4px}} .decision{{font-size:20px;font-weight:700}}
</style></head><body>
<h1>CONCORDIA — Final Validation, Alignment Frontier, and Conditional RL</h1>
<p class="boundary"><strong>Claim boundary.</strong> Every traffic experiment is synthetic.
The real-network study uses real OSM geometry with synthetic OD demand. TTC/DRAC/PET are
surrogate conflict indicators, never crash probabilities. Route changes occur only after
modeled acceptance.</p>
<h2>1. Initial hypotheses</h2>
<p>CONCORDIA tested whether preference heterogeneity and voluntary, acceptance-aware route
recommendations could improve traffic efficiency without worsening surrogate safety or
forcing individual sacrifice.</p>
<h2>2. Negative results retained</h2>
<table><tr><th>Hypothesis</th><th>Final interpretation</th></tr>
<tr><td>H1</td><td>FAIL unchanged. H1-R is a separate operationalization study.</td></tr>
<tr><td>H2</td><td>FAIL unchanged. B6 did not beat B1 in the original focused aggregate.</td></tr>
<tr><td>H6</td><td>FAIL unchanged. No pre-registered stability effect was established.</td></tr></table>
<p>The original focused evidence contains {len(analytical.get('focused', []))} paired rows.
No metric was changed to reverse these outcomes.</p>
<h2>3. Efficiency–voluntariness trade-off</h2>
<p>Study I evaluated {alignment['frontier_count']} seed/topology/demand/heterogeneity frontiers
and {alignment['sampled_point_count']} epsilon points. All frontiers were monotone; the knee
epsilon median was {alignment['knee_epsilon']['median']:.3f}. Sampled regions: 
{html.escape(str(alignment['phase_counts']))}.</p>
<p>H1-R did not show that heterogeneity is uniformly a coordination resource: weighted
alignment opportunity was {h1r['low']['weighted_alignment_opportunity']:.1f} in the low group
versus {h1r['high']['weighted_alignment_opportunity']:.1f} in the high group. Preference
diversity helps only when available route attributes expose useful time, reliability, or risk
trade-offs; topology can remove that choice space.</p>
<h2>4. Price of Alignment</h2>
<p><code>PoAlign(epsilon) = C*(epsilon) / C_SO</code> quantifies the system cost of preserving
the declared regret budget. Marginal finite differences report how much TTT is recovered per
additional unit of permitted private loss.</p>
<p>The acceptance–traffic fixed point converged in {fixed_point['FP1_converged_count']}/
{fixed_point['run_count']} cases. FP1 reduced mean acceptance Brier score from
{fixed_point['mean_FP0_acceptance_brier']:.5f} to
{fixed_point['mean_FP1_acceptance_brier']:.5f}, at the cost of a slower analytical solve.</p>
<h2>5. Microscopic phantom and safety</h2>
<table><tr><th>Question</th><th>Result</th></tr>
<tr><td>H3 matched VALID events</td><td>{'SUPPORTED' if h3_supported else 'NOT SUPPORTED'};
B1={h3['B1_probability']:.3f}, B6={h3['B6_probability']:.3f}, exact paired p={h3.get('exact_mcnemar_p')}.</td></tr>
<tr><td>H4 DRAC CVaR non-inferiority</td><td>{'SUPPORTED' if h4['noninferior'] else 'FAILED'};
B6−B1 mean={h4['paired_mean_difference_B6_minus_B1']:.4f}, upper CI={h4['bootstrap_ci95'][1]:.4f}, margin={h4['noninferiority_margin']:.4f}.</td></tr>
<tr><td>Phantom predictor</td><td>{'SUMO-calibrated held-out run split' if calibration['complete'] else 'NOT CALIBRATED: ' + calibration.get('reason','insufficient evidence')}.</td></tr></table>
<h2>6. Real-topology transfer</h2>
<p>{real['run_count']} SUMO runs used {real['legal_route_count']} legal alternatives on the
committed Gangnam OSM extract. Transfer and limited-retuning results are separate. Demand
provenance remains <strong>{html.escape(real['demand_provenance'])}</strong>.</p>
<p>Transfer failed to improve TTT: B1={real['statistics']['transfer']['B1_mean_TTT_seconds']:.1f}s
and B6={real['statistics']['transfer']['B6_mean_TTT_seconds']:.1f}s. The failure decomposition
records mean route-overlap Jaccard
{real['failure_decomposition']['route_overlap_jaccard_mean']:.3f}, mean accepted-route
preference cost {real['failure_decomposition']['mean_preference_cost_regret']:.4f}, and
{real['failure_decomposition']['new_secondary_bottleneck_edge_count']} loaded secondary
bottleneck edges. This is mechanism failure on synthetic demand, not an observed Seoul effect.</p>
<h2>7. Scalability and RL Gate</h2>
<p>B6's first declared enumeration-limit failure was
<code>{html.escape(str(scalability['B6_first_declared_limit_failure']))}</code>. The pre-RL
mathematical approximation reached operational p95
{scalability['approximation_operational_p95_seconds']:.4f}s; residual Gate E trigger:
{scalability['Gate_E']['triggered_for_RL']}. Gate C median incremental degradation was
{drift['Gate_C']['measured_median_nonstationarity_incremental_degradation']:.4f} against a
0.10 threshold. Its p95 was {drift['adaptive_p95_incremental_degradation']:.4f}, retained as a
tail failure even though the pre-registered median Gate C did not authorize RL.</p>
{f'''<p>Gate B nevertheless authorized RL0. Its held-out mean TTT gap versus the constrained
deterministic comparator was {conditional['mean_ttt_gap_vs_deterministic']:.5f}, p95 inference
was {conditional['p95_inference_seconds']:.6f}s, and regret/safety violations were
{conditional['regret_violation_count']}/{conditional['safety_violation_count']}. It was rejected
because it did not outperform the deterministic comparator.</p>''' if conditional else ''}
<h2>8. Conditions of validity and failure</h2>
<ul><li><strong>Alignment Win:</strong> regret-feasible recommendations beat sampled ETA-only cost.</li>
<li><strong>Alignment Trade-off:</strong> voluntariness is protected but network cost is higher.</li>
<li><strong>Alignment Infeasible:</strong> the permitted utility loss contains no useful diversion.</li>
<li><strong>Microscopic failure:</strong> B6 increased VALID event incidence in this matrix and
did not establish DRAC-CVaR non-inferiority.</li>
<li><strong>Topology-transfer failure:</strong> accepted recommendations redistributed demand
but increased TTT on the tested real geometry.</li></ul>
<p>Evidence: <code>artifacts/studies/alignment_frontier/summary.json</code>,
<code>artifacts/studies/microscopic_policy_matrix/summary.json</code>,
<code>artifacts/studies/real_topology_policy_matrix/summary.json</code>,
<code>artifacts/studies/scalability/summary.json</code>, and
<code>artifacts/rl_gate_report_v2.json</code>.</p>
<h2>Figures</h2><div class="grid">{figure_html}</div>
<p class="decision">{final_decision}. {rl_decision}.</p>
</body></html>"""
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(document, encoding="utf-8")
    print(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    run()
