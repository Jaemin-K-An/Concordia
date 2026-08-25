#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from v7_frozen import verify_frozen, write_json


ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str):
    return json.loads((ROOT / relative).read_text())


def _outcome(micro: dict, osm: dict) -> str:
    primary = micro["primary_metrics"]
    if primary["safety_violation_count"] > 0 or primary["deployment_precision"] < 0.60:
        return "Outcome F"
    if (
        primary["deployment_precision"] >= 0.80
        and primary["coverage"] >= 0.15
        and primary["opportunity_recovery_rate"] >= 0.50
        and primary["intervention_count"] >= 50
        and osm["primary_metrics"]["success_count"] > 0
    ):
        return "Outcome S+"
    if (
        primary["deployment_precision"] >= 0.80
        and primary["coverage"] >= 0.10
        and primary["opportunity_recovery_rate"] >= 0.40
        and primary["intervention_count"] >= 30
    ):
        return "Outcome S"
    return "Outcome P"


def _condition_effects(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    topology = defaultdict(list)
    penetration = defaultdict(list)
    for row in rows:
        value = float(row["outcomes"]["tau_t_relative"])
        topology[str(row["condition"]["topology"])].append(value)
        penetration[float(row["condition"]["penetration"])].append(value)
    topologies = [
        {
            "topology": name,
            "count": len(values),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "positive_rate": float(np.mean(np.asarray(values) >= 0.01)),
        }
        for name, values in sorted(topology.items())
    ]
    penetrations = [
        {
            "penetration": name,
            "count": len(values),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "positive_rate": float(np.mean(np.asarray(values) >= 0.01)),
        }
        for name, values in sorted(penetration.items())
    ]
    return topologies, penetrations


def _status(value: bool) -> str:
    return "PASS" if value else "FAIL"


def run() -> Path:
    verify_frozen()
    dataset = _read("artifacts/studies/v7_paired_dataset/dataset_summary.json")
    validation = _read("artifacts/studies/v7_policy_validation/validation_summary.json")
    micro = _read("artifacts/studies/v7_frozen_micro_holdout/summary.json")
    final_rows = _read("artifacts/studies/v7_frozen_micro_holdout/raw_metrics.json")
    analytical = _read("artifacts/studies/v7_frozen_analytical_holdout/summary.json")
    osm = _read("artifacts/studies/v7_real_topology/summary.json")
    failure = _read("artifacts/studies/v7_failure_analysis/summary.json")
    negative = _read("artifacts/studies/v7_failure_analysis/treatment_effect_by_condition.json")
    placebo = _read("artifacts/studies/v7_model_selection/placebo.json")
    holdouts = _read("artifacts/studies/v7_model_selection/scenario_family_holdouts.json")
    primary = micro["primary_metrics"]
    policies = micro["policy_metrics"]
    topology, penetration = _condition_effects(final_rows)
    outcome = _outcome(micro, osm)
    h = {
        "H37": (
            primary["deployment_precision"] >= 0.80
            and primary["coverage"] > policies["V6-Binary"]["coverage"]
        ),
        "H38": policies["V7_quantile"]["deployment_precision"]
        > policies["V7_mean"]["deployment_precision"],
        "H39": policies["V7-TS"]["safety_violation_count"]
        < policies["V7-T"]["safety_violation_count"],
        "H40": primary["opportunity_recovery_rate"]
        > policies["V6-Binary"]["opportunity_recovery_rate"],
        "H41": primary["deployment_precision"] >= 0.80
        and primary["intervention_count"] > 0,
        "H42": primary["coverage"] >= 0.10,
        "H43": primary["safety_violation_count"] == 0,
        "H44": osm["primary_metrics"]["success_count"] >= 1,
    }
    negative_conditions = [
        row for row in negative if row["structurally_negative_descriptive"]
    ]
    negative_labels = ", ".join(
        f"{row['axis']}={row['value']}" for row in negative_conditions
    )
    topology_lines = "\n".join(
        f"| {row['topology']} | {row['count']} | {row['mean']:.4f} | {row['median']:.4f} | {row['positive_rate']:.3f} |"
        for row in topology
    )
    penetration_lines = "\n".join(
        f"| {row['penetration']:.2f} | {row['count']} | {row['mean']:.4f} | {row['median']:.4f} | {row['positive_rate']:.3f} |"
        for row in penetration
    )
    h_lines = "\n".join(
        f"| {name} | **{_status(value)}** |"
        for name, value in h.items()
    )
    report = f"""# CONCORDIA v7 Final Research Report

## Result

**{outcome}**

CONCORDIA v7 changed the primary question from binary success prediction to paired conditional
treatment-effect estimation. The frozen policy intervenes only when the traffic-effect lower
bound exceeds 1%, the safety-effect upper bound is at most 0.25, predicted maximum regret is at
most 0.08, and the route is legal. SafeMicroSuccess is retained only as the deployment precision
label. No final microscopic or OSM result was used to tune a model, interval, or threshold.

The development corpus contains **{dataset['pair_count']} paired cases**, of which
**{dataset['new_actual_sumo_pair_count']}** are newly generated v7 actual-SUMO cases. The final
microscopic holdout contains **{micro['pair_count']} fresh paired cases / {micro['actual_sumo_run_count']}
SUMO runs**. V7-F made **{primary['intervention_count']} interventions**, with deployment precision
**{primary['deployment_precision']:.3f}**, coverage **{primary['coverage']:.3f}**, ORR
**{primary['opportunity_recovery_rate']:.3f}**, and **{primary['safety_violation_count']}** safety
violations. The 95% Wilson precision lower bound is **{primary['precision_wilson_95_lower']:.3f}**.

## Ten required research questions

1. **Why was binary classification insufficient?** v6 collapses benefit magnitude, safety effect,
   and regret into one label. On the same v7 final holdout V6-Binary recovered
   {policies['V6-Binary']['opportunity_recovery_rate']:.3f} of safe opportunities, while the
   continuous model ranked traffic effects at Spearman
   {micro['traffic_effect_metrics']['spearman']:.3f} and separately exposed poor safety-effect
   transfer. That magnitude information was scientifically useful, but V7-F also recovered zero
   opportunities: replacing the binary target did not by itself solve safe deployment.

2. **How predictable was the continuous paired effect?** Final PEHE-like traffic-effect MAE was
   **{micro['traffic_effect_metrics']['mae']:.4f}**, RMSE **{micro['traffic_effect_metrics']['rmse']:.4f}**,
   and Spearman correlation **{micro['traffic_effect_metrics']['spearman']:.3f}**. These are direct
   errors against paired counterfactual effects, not observational causal proxies.

3. **How often was the traffic-uplift sign correct?** Final sign accuracy was
   **{micro['traffic_effect_metrics']['sign_accuracy']:.3f}** and positive-uplift recall was
   **{micro['traffic_effect_metrics']['positive_uplift_recall']:.3f}**.

4. **How stable was safety-effect prediction?** Final safety-effect MAE was
   **{micro['safety_effect_metrics']['mae']:.4f}**, RMSE **{micro['safety_effect_metrics']['rmse']:.4f}**,
   and the all-case false-safe rate under the frozen safety UCB was
   **{micro['safety_effect_false_safe_rate']:.3f}**.

5. **Did interval selection improve precision over mean selection?** On the identical final
   holdout, V7-mean precision was **{policies['V7_mean']['deployment_precision']:.3f}**,
   bootstrap-quantile precision **{policies['V7_quantile']['deployment_precision']:.3f}**, and
   conformal precision **{policies['V7_conformal']['deployment_precision']:.3f}**. H38 is reported
   strictly from these frozen comparisons; zero-intervention variants are not credited with
   perfect precision.

6. **Did treatment-effect selection improve ORR over v6?** V7-F ORR was
   **{primary['opportunity_recovery_rate']:.3f}** versus V6-Binary
   **{policies['V6-Binary']['opportunity_recovery_rate']:.3f}** on the same final cases.

7. **What did high navigation penetration do?** The table below reports paired effects rather
   than assuming monotonic improvement. At 100% penetration the observed mean effect was
   **{next(row['mean'] for row in penetration if row['penetration'] == 1.0):.4f}**; comparison with
   lower penetration is descriptive within the synthetic randomized design.

8. **How did effects differ by topology?** The topology table shows substantial heterogeneity;
   it is the evidence used for this answer, including unsuccessful topologies rather than
   filtering them after the fact.

9. **Was positive uplift identified on real OSM geometry?** The frozen policy made
   **{osm['primary_metrics']['intervention_count']}** interventions and recovered
   **{osm['primary_metrics']['success_count']}** safe beneficial cases across
   **{osm['od_pair_count']}** prespecified Gangnam OD pairs, despite
   **{osm['primary_metrics']['opportunity_count']}** paired counterfactual safe opportunities.
   Demand and preferences remain
   synthetic; this is not an observed-Seoul causal claim.

10. **Where was Adaptive structurally negative?** The condition analysis found
    **{len(negative_conditions)}** descriptive axis levels with negative mean paired effect.
    They were `{negative_labels}`. These regimes are preserved in the failure artifact;
    plausible mechanisms include secondary
    bottlenecks, partial adoption mismatch, and topology transfer, but the mechanism labels are
    diagnostics rather than separately identified causal effects.

## Final policy comparison

| Policy | Interventions | Precision | Coverage | ORR | Safety violations |
|---|---:|---:|---:|---:|---:|
| B6 always-on | {policies['B6']['intervention_count']} | {policies['B6']['deployment_precision']:.3f} | {policies['B6']['coverage']:.3f} | {policies['B6']['opportunity_recovery_rate']:.3f} | {policies['B6']['safety_violation_count']} |
| V5-F | {policies['V5-F']['intervention_count']} | {policies['V5-F']['deployment_precision']:.3f} | {policies['V5-F']['coverage']:.3f} | {policies['V5-F']['opportunity_recovery_rate']:.3f} | {policies['V5-F']['safety_violation_count']} |
| V6-Binary | {policies['V6-Binary']['intervention_count']} | {policies['V6-Binary']['deployment_precision']:.3f} | {policies['V6-Binary']['coverage']:.3f} | {policies['V6-Binary']['opportunity_recovery_rate']:.3f} | {policies['V6-Binary']['safety_violation_count']} |
| V7-T | {policies['V7-T']['intervention_count']} | {policies['V7-T']['deployment_precision']:.3f} | {policies['V7-T']['coverage']:.3f} | {policies['V7-T']['opportunity_recovery_rate']:.3f} | {policies['V7-T']['safety_violation_count']} |
| V7-TS | {policies['V7-TS']['intervention_count']} | {policies['V7-TS']['deployment_precision']:.3f} | {policies['V7-TS']['coverage']:.3f} | {policies['V7-TS']['opportunity_recovery_rate']:.3f} | {policies['V7-TS']['safety_violation_count']} |
| V7-TSU | {policies['V7-TSU']['intervention_count']} | {policies['V7-TSU']['deployment_precision']:.3f} | {policies['V7-TSU']['coverage']:.3f} | {policies['V7-TSU']['opportunity_recovery_rate']:.3f} | {policies['V7-TSU']['safety_violation_count']} |
| **V7-F** | **{primary['intervention_count']}** | **{primary['deployment_precision']:.3f}** | **{primary['coverage']:.3f}** | **{primary['opportunity_recovery_rate']:.3f}** | **{primary['safety_violation_count']}** |

## Effect by penetration

| Penetration | N | Mean effect | Median effect | Positive ≥1% |
|---:|---:|---:|---:|---:|
{penetration_lines}

## Effect by topology

| Topology | N | Mean effect | Median effect | Positive ≥1% |
|---|---:|---:|---:|---:|
{topology_lines}

## Hypotheses H37–H44

| Hypothesis | Status |
|---|---:|
{h_lines}

H37 uses Coverage@Precision80 on the same final holdout. H38 compares frozen mean and quantile
variants. H39 compares V7-T with V7-TS safety violations. H40 compares ORR with V6-Binary on the
same cases. H41–H44 apply the preregistered final thresholds literally.

## Robustness and diagnostics

- Development validation selected `{validation['selected_traffic']['key']}` and
  `{validation['selected_policy']['method']}` uncertainty handling.
- The paired placebo real-signal Spearman was
  {placebo['real_signal']['spearman']:.3f}, versus
  {placebo['permuted_target']['spearman']:.3f} after target permutation.
- Three development-only family holdouts were executed: {', '.join(row['holdout'] for row in holdouts)}.
- Historical analytical V5-RD precision on the 512-case correctness check was
  {analytical['historical_v5_rd_reference_metrics']['intervention_precision']:.3f}.
- Failure analysis retained {failure['false_positive_count']} false positives and
  {failure['false_negative_count']} missed safe opportunities without topology exclusion.

## Claim boundary

The paired treatment effects are identified inside randomized synthetic SUMO experiments. The
OSM study preserves real road geometry but uses synthetic OD demand and modeled preferences.
TTC, PET, and DRAC are surrogate conflict indicators, not crash outcomes. CONCORDIA changes only
legal route recommendations after modeled acceptance; it does not control vehicle motion. RL was
not used because the v7 question is conditional-effect estimation and conservative selection.
"""
    path = ROOT / "reports/v7_final_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    write_json(
        ROOT / "artifacts/reports/v7_final_report_summary.json",
        {
            "complete": True,
            "outcome": outcome,
            "hypotheses": h,
            "primary_metrics": primary,
            "topology_effects": topology,
            "penetration_effects": penetration,
            "negative_conditions": negative_conditions,
            "rl_used": False,
        },
    )
    print(path)
    return path


if __name__ == "__main__":
    run()
