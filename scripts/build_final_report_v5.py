#!/usr/bin/env python3
from __future__ import annotations

import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text())


def _metric_cards(a: dict, s: dict, m: dict, r: dict) -> str:
    cards = [
        ("Analytical precision", f"{a['intervention_precision']:.1%}", "target ≥80% · pass"),
        ("Analytical coverage", f"{a['coverage']:.1%}", "target ≥15% · fail"),
        ("Stress precision", f"{s['intervention_precision']:.1%}", "target ≥70% · pass"),
        ("Micro precision", f"{m['intervention_precision']:.1%}", "target >50% · fail"),
        ("Micro false-safe", f"{m['false_safe_rate']:.1%}", "target ≤5% · fail"),
        ("Real OSM activation", f"{r['coverage']:.1%}", "48 paired conditions · all abstain"),
    ]
    return "".join(
        f'<div class="metric"><span>{html.escape(label)}</span><strong>{value}</strong><small>{html.escape(note)}</small></div>'
        for label, value, note in cards
    )


def run() -> Path:
    if not (ROOT / "artifacts/v5/final_audit.json").is_file():
        from run_v5_final_audit import run as run_audit

        run_audit()
    audit = _load("artifacts/v5/final_audit.json")
    analytical = _load("artifacts/studies/v5_frozen_holdout/summary.json")
    stress = _load("artifacts/studies/v5_stress_holdout/summary.json")
    micro = _load("artifacts/studies/v5_microscopic_holdout/summary.json")
    real = _load("artifacts/studies/v5_real_topology/summary.json")
    a = analytical["primary_metrics"]
    s = stress["primary_metrics"]
    m = micro["primary_metrics"]
    r = real["primary_metrics"]
    hypothesis_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{html.escape(value['status'])}</td><td>{html.escape(value['finding'])}</td></tr>"
        for name, value in audit["hypotheses"].items()
    )
    html_report = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CONCORDIA v5 — Final Research Report</title>
<style>
:root{{--ink:#17231f;--muted:#5b6a65;--paper:#f5f2ea;--panel:#fffdf8;--green:#0b6e4f;--red:#a43f32;--line:#d7d5cb}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.62 Inter,ui-sans-serif,system-ui,sans-serif}}
main{{max-width:1120px;margin:auto;padding:64px 28px 96px}} header{{border-top:8px solid var(--red);padding:54px 0 42px;border-bottom:1px solid var(--line)}}
.eyebrow{{letter-spacing:.16em;text-transform:uppercase;color:var(--red);font-weight:750;font-size:.78rem}} h1{{font-family:Georgia,serif;font-size:clamp(3rem,8vw,6.7rem);line-height:.9;margin:.3rem 0 1.2rem;letter-spacing:-.055em}}
.dek{{font-family:Georgia,serif;font-size:1.4rem;max-width:780px;color:var(--muted)}} .outcome{{display:inline-flex;gap:16px;align-items:center;margin-top:24px;padding:14px 18px;background:#f3dfdb;border-left:5px solid var(--red)}}
.outcome b{{font-size:2rem;color:var(--red)}} section{{padding:46px 0;border-bottom:1px solid var(--line)}} h2{{font-family:Georgia,serif;font-size:2.2rem;margin:0 0 18px}} h3{{margin-top:28px}}
.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}} .metric{{background:var(--panel);border:1px solid var(--line);padding:19px;display:flex;flex-direction:column;min-height:144px}}
.metric span,.metric small{{color:var(--muted)}} .metric strong{{font:2.3rem Georgia,serif;margin:auto 0 .1rem}} table{{width:100%;border-collapse:collapse;background:var(--panel)}} th,td{{padding:12px 14px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}} th{{font-size:.76rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}}
.pipeline{{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin:24px 0}} .pipeline div{{background:var(--ink);color:white;padding:15px;min-height:92px}} .pipeline b{{display:block;color:#81c7ad}}
.note{{border-left:4px solid var(--green);background:#e5efe9;padding:16px 20px}} .failure{{border-color:var(--red);background:#f3dfdb}}
code{{background:#ebe8df;padding:.1em .35em}} footer{{padding-top:36px;color:var(--muted);font-size:.9rem}} @media(max-width:760px){{.grid{{grid-template-columns:1fr 1fr}}.pipeline{{grid-template-columns:1fr}}}} @media(max-width:480px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><main>
<header><div class="eyebrow">Robust regime-conditioned selective adaptive navigation</div><h1>CONCORDIA<br>v5</h1>
<p class="dek">A frozen, leakage-resistant evaluation of regime conditioning, domain-shift abstention, microscopic correction, and a surrogate-safety veto.</p>
<div class="outcome"><b>Outcome {audit['outcome']}</b><span>Analytical precision passed. Microscopic safety failed and independently forces rejection.</span></div></header>
<section><h2>Decision at a glance</h2><div class="grid">{_metric_cards(a,s,m,r)}</div>
<p class="note failure"><strong>No deployment claim.</strong> The full actual-SUMO policy made {m['intervention_count']} interventions, achieved {m['successful_intervention_count']} success, and allowed {m['safety_violation_count']} surrogate safety violation. The real-geometry study abstained in all {real['paired_condition_count']} conditions.</p></section>
<section><h2>Frozen research sequence</h2><div class="pipeline"><div><b>01</b>Regime discovery</div><div><b>02</b>Shift detector</div><div><b>03</b>Micro correction</div><div><b>04</b>Five-package freeze</div><div><b>05</b>Untouched holdouts</div></div>
<p>The manifest fixed 104 deployment-code checksums and nine model artifacts before any final analytical, stress, microscopic, or OSM holdout was materialized. Final analytical N={analytical['case_count']}; final microscopic paired N={micro['pair_count']}.</p></section>
<section><h2>Primary results</h2><table><thead><tr><th>Domain</th><th>Cases</th><th>Interventions</th><th>Precision</th><th>Coverage</th><th>Safety violations</th></tr></thead><tbody>
<tr><td>Synthetic analytical</td><td>{analytical['case_count']}</td><td>{a['intervention_count']}</td><td>{a['intervention_precision']:.4f}</td><td>{a['coverage']:.4f}</td><td>{a['safety_violation_count']}</td></tr>
<tr><td>Unseen stress</td><td>{stress['case_count']}</td><td>{s['intervention_count']}</td><td>{s['intervention_precision']:.4f}</td><td>{s['coverage']:.4f}</td><td>{s['safety_violation_count']}</td></tr>
<tr><td>Actual SUMO microscopic</td><td>{micro['pair_count']}</td><td>{m['intervention_count']}</td><td>{m['intervention_precision']:.4f}</td><td>{m['coverage']:.4f}</td><td>{m['safety_violation_count']}</td></tr>
<tr><td>Real OSM geometry</td><td>{real['paired_condition_count']}</td><td>{r['intervention_count']}</td><td>{r['intervention_precision']:.4f}</td><td>{r['coverage']:.4f}</td><td>{r['safety_violation_count']}</td></tr></tbody></table>
<p>V5-RD met analytical precision and intervention-count targets, plus the 0.70 worst-critical-group precision target ({analytical['group_metrics']['worst_critical_group_precision']:.2f}), but missed 15% coverage. Stress precision remained above 70%.</p></section>
<section><h2>Hypotheses H21–H28</h2><table><thead><tr><th>Hypothesis</th><th>Status</th><th>Frozen finding</th></tr></thead><tbody>{hypothesis_rows}</tbody></table></section>
<section><h2>Failure taxonomy</h2><div class="grid">
<div class="metric"><span>Benefit prediction error</span><strong>{micro['failure_taxonomy'].get('benefit_prediction_error',0)}</strong><small>final micro pairs</small></div>
<div class="metric"><span>Safety mismatch</span><strong>{micro['failure_taxonomy'].get('microscopic_safety_mismatch',0)}</strong><small>surrogate DRAC tail</small></div>
<div class="metric"><span>Partial adoption feedback</span><strong>{micro['failure_taxonomy'].get('partial_adoption_feedback',0)}</strong><small>acceptance below 0.5</small></div></div>
<p>Microscopic correction did not improve benefit MAE: analytical {micro['hypotheses']['H24_micro_correction_reduces_benefit_mae']['analytical_mae']:.5f}, corrected {micro['hypotheses']['H24_micro_correction_reduces_benefit_mae']['corrected_mae']:.5f}. DSS did not beat the no-DSS stress ablation.</p></section>
<section><h2>Aggregation disclosure</h2><p class="note">The frozen microscopic summary assigned B6 TTT deltas to abstained rows in one descriptive aggregation. Decisions, precision, coverage, success, and safety labels are unaffected. The immutable raw-pair recomputation gives selected-policy population mean TTT gain <strong>{audit['posthoc_metric_correction']['corrected_population_mean_network_ttt_gain_seconds']:.4f} s</strong>; no policy or threshold changed.</p></section>
<section><h2>Evidence boundary</h2><ul><li>Analytical results use synthetic BPR cases.</li><li>Microscopic results use actual SUMO with synthetic demand/preferences.</li><li>OSM contributes real road geometry, not observed Seoul demand.</li><li>TTC, PET, and DRAC are surrogate conflict measures—not crash probabilities.</li><li>RL was excluded by the v5 preregistration.</li></ul></section>
<footer>CONCORDIA v5 · frozen source {html.escape(audit['freeze_source_commit'][:12])} · full evidence in FINAL_AUDIT_V5.md and artifacts/studies/v5_*</footer>
</main></body></html>"""
    output = ROOT / "artifacts/report.html"
    output.write_text(html_report, encoding="utf-8")
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "v5_methodology.md").write_text(
        """# CONCORDIA v5 Methodology

v5 separates development into analytical training, calibration, validation, shift validation,
micro development, micro calibration, and micro validation. Final analytical, stress,
microscopic, and OSM seeds are disjoint. A learned regime router uses navigation penetration and
structural route features; a robust median/MAD shift detector produces DSS and three shift cells.
The analytical primary policy is V5-RD. The actual-SUMO bridge adds a benefit correction,
microscopic success calibration, and bootstrap-logistic safety UCB veto.

Five deployment YAML packages and a SHA-256 manifest were committed before holdout generation.
Strong shift always abstains. The calibration protocol is fixed to ten equal-width bins on [0,1].
All success labels require legal execution, bounded regret/safety, and at least 1% relative TTT gain.
RL is excluded.
""",
        encoding="utf-8",
    )
    (docs / "v5_failure_analysis.md").write_text(
        f"""# CONCORDIA v5 Failure Analysis

The analytical primary policy passed 80% precision but missed 15% coverage. The stress target
passed, although DSS did not improve precision over the no-DSS ablation. The micro correction
worsened final benefit MAE from
{micro['hypotheses']['H24_micro_correction_reduces_benefit_mae']['analytical_mae']:.6f} to
{micro['hypotheses']['H24_micro_correction_reduces_benefit_mae']['corrected_mae']:.6f}.

In 100 actual-SUMO final pairs, V5-F intervened 10 times, succeeded once, and admitted one
surrogate-safety failure. False-safe rate was 10%. On six stratified OSM OD pairs and 48 paired
conditions it abstained everywhere. The next version should treat microscopic safety and benefit
as first-class causal targets, enlarge seed-disjoint micro development data, and require a
nonzero activation validation guard before freeze.
""",
        encoding="utf-8",
    )
    (docs / "v5_deployment_checklist.md").write_text(
        """# CONCORDIA v5 Deployment Checklist

**Deployment is blocked.** Before any field interpretation:

- [ ] Actual-SUMO intervention precision exceeds 0.50 on a new holdout.
- [ ] Actual-SUMO safety violations are zero and false-safe rate is at most 0.05.
- [ ] At least ten microscopic interventions and one safe success are observed.
- [ ] Real-geometry activation is nonzero on at least six legal OD pairs.
- [ ] Analytical coverage reaches 0.15 while retaining 0.80 precision.
- [ ] The microscopic aggregation defect is fixed before a new freeze.
- [ ] External traffic counts, calibrated demand, and independent safety review are added.

Passing analytical or surrogate checks alone never authorizes deployment.
""",
        encoding="utf-8",
    )
    print(output)
    return output


if __name__ == "__main__":
    run()
