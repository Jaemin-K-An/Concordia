#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from v5_frozen import ROOT, verify_frozen, write_json


def _load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text())


def _status(value: bool) -> str:
    return "PASS" if value else "FAIL"


def run() -> Path:
    manifest = verify_frozen()
    analytical = _load("artifacts/studies/v5_frozen_holdout/summary.json")
    stress = _load("artifacts/studies/v5_stress_holdout/summary.json")
    micro = _load("artifacts/studies/v5_microscopic_holdout/summary.json")
    real = _load("artifacts/studies/v5_real_topology/summary.json")
    development = _load(
        "artifacts/studies/v5_policy_validation/hypothesis_development_results.json"
    )
    micro_rows = json.loads(
        (ROOT / "artifacts/studies/v5_microscopic_holdout/raw_metrics.json").read_text()
    )
    selected_gain = np.asarray(
        [
            row["b1"]["total_travel_time_seconds"]
            - row["b6"]["total_travel_time_seconds"]
            if row["intervene"]
            else 0.0
            for row in micro_rows
        ],
        dtype=float,
    )
    selected_relative_gain = np.asarray(
        [
            row["microscopic_benefit"] if row["intervene"] else 0.0
            for row in micro_rows
        ],
        dtype=float,
    )
    analytical_metrics = analytical["primary_metrics"]
    micro_metrics = micro["primary_metrics"]
    safety_failure = micro_metrics["safety_violation_count"] > 0
    if analytical_metrics["intervention_precision"] < 0.80 or safety_failure:
        outcome = "F"
    elif (
        analytical_metrics["coverage"] >= 0.20
        and stress["stress_target_met"]
        and micro["claim_eligible"]
    ):
        outcome = "S+"
    elif analytical_metrics["coverage"] >= 0.15 and micro["claim_eligible"]:
        outcome = "S"
    else:
        outcome = "P"
    policy_metrics = analytical["policy_metrics"]
    hypotheses = {
        "H21_regime_conditioning": {
            "status": _status(
                policy_metrics["V5-R"]["coverage"] > policy_metrics["V5-G"]["coverage"]
                and policy_metrics["V5-R"]["intervention_precision"]
                >= policy_metrics["V5-G"]["intervention_precision"]
            ),
            "finding": "Frozen holdout V5-R slightly improved both precision and coverage over V5-G.",
        },
        "H22_DSS": {
            "status": _status(stress["H22_DSS_improves_shift_safety"]["supported"]),
            "finding": "DSS retained zero safety violations but did not improve stress precision over the no-DSS ablation.",
        },
        "H23_micro_safety_gate": {
            "status": "FAIL / OVER-CONSERVATIVE",
            "finding": "The safety gate reduced analytical activation to zero and still allowed one microscopic false-safe intervention.",
        },
        "H24_micro_correction": {
            "status": _status(
                micro["hypotheses"]["H24_micro_correction_reduces_benefit_mae"][
                    "corrected_mae"
                ]
                < micro["hypotheses"]["H24_micro_correction_reduces_benefit_mae"][
                    "analytical_mae"
                ]
            ),
            "finding": "Microscopic benefit correction slightly worsened final-holdout MAE.",
        },
        "H25_safe_micro_success": {
            "status": _status(micro["claim_eligible"]),
            "finding": "There was one safe success, but precision was 0.10 and one safety violation occurred.",
        },
        "H26_selectivity_mechanism": {
            "status": "PARTIAL",
            "finding": "Selectivity removed all analytical safety violations and 24/25 microscopic unsafe adaptations, but not the last false-safe case.",
        },
        "H27_hierarchical_or_mixture": {
            "status": "FAIL",
            "finding": "Model selection chose regime-specific M3, not the hierarchical or mixture candidates.",
        },
        "H28_penetration_interactions": {
            "status": _status(development["H28"]["supported_on_validation"]),
            "finding": "The interaction model did not jointly improve validation coverage and Brier score over its nested global comparator.",
        },
    }
    checks = {
        "freeze_manifest_verified": True,
        "post_freeze_deployment_code_unchanged": True,
        "analytical_case_count_at_least_800": analytical["case_count"] >= 800,
        "analytical_interventions_at_least_75": analytical_metrics["intervention_count"] >= 75,
        "analytical_precision_at_least_0_80": analytical_metrics[
            "intervention_precision"
        ]
        >= 0.80,
        "analytical_coverage_at_least_0_15": analytical_metrics["coverage"] >= 0.15,
        "critical_group_precision_at_least_0_70": analytical["group_metrics"][
            "worst_critical_group_precision"
        ]
        >= 0.70,
        "stress_target_met": stress["stress_target_met"],
        "microscopic_pair_count_at_least_100": micro["pair_count"] >= 100,
        "microscopic_interventions_at_least_10": micro_metrics["intervention_count"]
        >= 10,
        "microscopic_precision_above_0_50": micro_metrics["intervention_precision"]
        > 0.50,
        "microscopic_safe_success_positive": micro_metrics["safe_success_count"] > 0,
        "microscopic_safety_violations_zero": micro_metrics["safety_violation_count"]
        == 0,
        "microscopic_false_safe_at_most_0_05": micro_metrics["false_safe_rate"]
        <= 0.05,
        "real_topology_od_count_between_6_and_10": 6 <= real["od_pair_count"] <= 10,
        "real_topology_routes_legal": real["all_routes_legal"],
        "rl_excluded": not any(
            summary.get("rl_used", False)
            for summary in (analytical, stress, micro, real)
        ),
    }
    audit = {
        "complete": True,
        "version": "CONCORDIA v5",
        "outcome": outcome,
        "outcome_reason": (
            "Final microscopic safety failure forces Outcome F despite analytical precision and stress precision passing."
            if outcome == "F"
            else "Outcome follows preregistered analytical, stress, and microscopic gates."
        ),
        "checks": checks,
        "check_counts": {
            "passed": sum(checks.values()),
            "failed": sum(not value for value in checks.values()),
        },
        "analytical_primary": analytical_metrics,
        "stress_primary": stress["primary_metrics"],
        "microscopic_primary": micro_metrics,
        "real_topology_primary": real["primary_metrics"],
        "hypotheses": hypotheses,
        "posthoc_metric_correction": {
            "scope": "reporting aggregation only; frozen decisions and success/safety labels unchanged",
            "issue": "The frozen microscopic summary populated system_ttt_gain for abstentions before summarize_selective_policy.",
            "reported_mean_network_ttt_gain_seconds": micro_metrics[
                "mean_network_ttt_gain"
            ],
            "corrected_population_mean_network_ttt_gain_seconds": float(
                selected_gain.mean()
            ),
            "corrected_population_mean_relative_ttt_gain": float(
                selected_relative_gain.mean()
            ),
            "policy_or_threshold_changed": False,
        },
        "freeze_source_commit": manifest["source_commit"],
        "claim_boundaries": {
            "analytical": analytical["claim_boundary"],
            "microscopic": micro["claim_boundary"],
            "real_topology": real["claim_boundary"],
        },
    }
    output = ROOT / "artifacts/v5/final_audit.json"
    write_json(output, audit)
    markdown = f"""# CONCORDIA v5 Final Audit

## Decision

**Outcome {outcome}.** {audit['outcome_reason']}

The deployment code and five frozen YAML packages match the pre-holdout SHA-256 manifest. No
final seed entered model fitting, regime discovery, shift fitting, calibration, micro correction,
safety-veto fitting, or threshold selection.

## Primary evidence

| Domain | N / interventions | Precision | Coverage | Safety violations | Decision |
|---|---:|---:|---:|---:|---|
| Analytical | {analytical['case_count']} / {analytical_metrics['intervention_count']} | {analytical_metrics['intervention_precision']:.4f} | {analytical_metrics['coverage']:.4f} | {analytical_metrics['safety_violation_count']} | precision pass; coverage fail |
| Stress | {stress['case_count']} / {stress['primary_metrics']['intervention_count']} | {stress['primary_metrics']['intervention_precision']:.4f} | {stress['primary_metrics']['coverage']:.4f} | {stress['primary_metrics']['safety_violation_count']} | stress target pass |
| Actual SUMO microscopic | {micro['pair_count']} / {micro_metrics['intervention_count']} | {micro_metrics['intervention_precision']:.4f} | {micro_metrics['coverage']:.4f} | {micro_metrics['safety_violation_count']} | claim forbidden |
| Real OSM geometry | {real['paired_condition_count']} / {real['primary_metrics']['intervention_count']} | {real['primary_metrics']['intervention_precision']:.4f} | {real['primary_metrics']['coverage']:.4f} | {real['primary_metrics']['safety_violation_count']} | all abstain |

Analytical precision passed 0.80 with 132 interventions and zero analytical safety violations,
but coverage was 0.1289 rather than 0.15. Stress precision was 0.8519. In actual SUMO, the
full policy made 10 interventions, achieved one success, and allowed one surrogate safety
violation; false-safe rate was 0.10. That microscopic safety failure independently forces F.

## H21–H28

"""
    for name, value in hypotheses.items():
        markdown += f"- **{name}: {value['status']}** — {value['finding']}\n"
    markdown += f"""

## Transparent aggregation correction

The frozen microscopic evaluator incorrectly included B6 TTT deltas for abstained rows when
computing its descriptive population mean. The stored decisions, intervention count, precision,
success labels, and safety labels are unaffected. Recomputed from immutable raw pairs, the
selected policy's population-mean network TTT gain is
**{audit['posthoc_metric_correction']['corrected_population_mean_network_ttt_gain_seconds']:.4f} s**
(relative **{audit['posthoc_metric_correction']['corrected_population_mean_relative_ttt_gain']:.6f}**).
The original value remains preserved in the raw summary and no post-freeze threshold changed.

## Claim boundary

- Analytical evidence is synthetic and uses a BPR correctness harness.
- Microscopic evidence is actual SUMO with synthetic demand and preferences.
- OSM supplies real geometry only; its OD demand is synthetic.
- TTC, PET, and DRAC are surrogate conflict indicators, never crash probabilities.
- RL remained excluded because the v5 gate did not authorize it.
"""
    (ROOT / "FINAL_AUDIT_V5.md").write_text(markdown, encoding="utf-8")
    print(output)
    return output


if __name__ == "__main__":
    run()
