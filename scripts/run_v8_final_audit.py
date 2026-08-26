#!/usr/bin/env python3
from __future__ import annotations

import json

from v8_common import ROOT, write_json
from v8_frozen import verify_frozen


def run():
    manifest = verify_frozen()
    dataset = json.loads((ROOT / "artifacts/studies/v8_safety_dataset/dataset_summary.json").read_text())
    validation = json.loads((ROOT / "artifacts/studies/v8_policy_validation/summary.json").read_text())
    state_action = json.loads((ROOT / "artifacts/studies/v8_safety_model_selection/state_vs_action.json").read_text())
    final = json.loads((ROOT / "artifacts/studies/v8_micro_holdout/summary.json").read_text())
    osm = json.loads((ROOT / "artifacts/studies/v8_real_topology/summary.json").read_text())
    metrics = final["comparison"]["V8-F"]
    safety = final["safety_classifier"]
    cost = final["safety_cost_of_uplift"]
    rank = final["traffic_ranking"]
    checks = [
        ("development pairs >= 2000", dataset["pair_count"], dataset["pair_count"] >= 2000),
        ("unsafe development pairs >= 300", dataset["unsafe_intervention_count"], dataset["unsafe_intervention_count"] >= 300),
        ("new final pairs >= 400", final["pair_count"], final["pair_count"] >= 400),
        ("future-state leakage = 0", dataset["future_state_leakage_count"] + final["future_state_leakage_count"], dataset["future_state_leakage_count"] + final["future_state_leakage_count"] == 0),
        ("pairing failures = 0", dataset["pairing_failure_count"] + final["pairing_failure_count"], dataset["pairing_failure_count"] + final["pairing_failure_count"] == 0),
        ("traffic top-10% mean uplift positive", rank["top_k"]["top_10_percent"]["mean_realized_uplift"], rank["top_k"]["top_10_percent"]["mean_realized_uplift"] > 0),
        ("safety classifier PR-AUC reported", safety["pr_auc_average_precision"], safety["pr_auc_average_precision"] > 0),
        ("unsafe recall >= 0.95", safety["unsafe_recall"], safety["unsafe_recall"] >= 0.95),
        ("false-safe rate <= 0.05", safety["false_safe_rate_all_candidates"], safety["false_safe_rate_all_candidates"] <= 0.05),
        ("worst critical-group recall >= 0.85", safety["worst_critical_group_recall"], safety["worst_critical_group_recall"] >= 0.85),
        ("deployment precision >= 0.80", metrics["deployment_precision"], metrics["deployment_precision"] >= 0.80),
        ("coverage >= 0.08", metrics["coverage"], metrics["coverage"] >= 0.08),
        ("ORR >= 0.35", metrics["opportunity_realization_rate"], metrics["opportunity_realization_rate"] >= 0.35),
        ("interventions >= 30", metrics["intervention_count"], metrics["intervention_count"] >= 30),
        ("safety violations = 0", metrics["safety_violation_count"], metrics["safety_violation_count"] == 0),
        ("safe-success retention >= 0.70", cost["safe_success_retention"], cost["safe_success_retention"] >= 0.70),
        ("OSM interventions > 0", osm["primary_metrics"]["intervention_count"], osm["primary_metrics"]["intervention_count"] > 0),
        ("OSM safe interventions > 0", osm["primary_metrics"]["success_count"], osm["primary_metrics"]["success_count"] > 0),
        ("action-aware beats state-only PR-AUC", state_action["action_aware_beats_state_only_pr_auc"], state_action["action_aware_beats_state_only_pr_auc"]),
        ("action-aware beats state-only false-safe rate", state_action["action_aware_beats_state_only_false_safe_rate"], state_action["action_aware_beats_state_only_false_safe_rate"]),
        ("RL decision = NO", "NO", not final["rl_used"] and not osm["rl_used"]),
    ]
    if validation["safe_abstention"] or metrics["safety_violation_count"] > 0 or metrics["deployment_precision"] < 0.70:
        outcome = "F"
    elif (
        metrics["deployment_precision"] >= 0.80
        and metrics["coverage"] >= 0.10
        and metrics["opportunity_realization_rate"] >= 0.50
        and metrics["safety_violation_count"] == 0
        and metrics["intervention_count"] >= 40
        and osm["primary_metrics"]["success_count"] > 0
    ):
        outcome = "S+"
    elif (
        metrics["deployment_precision"] >= 0.80
        and metrics["coverage"] >= 0.08
        and metrics["opportunity_realization_rate"] >= 0.35
        and metrics["safety_violation_count"] == 0
        and metrics["intervention_count"] >= 30
    ):
        outcome = "S"
    elif metrics["deployment_precision"] >= 0.70 and metrics["safety_violation_count"] == 0 and metrics["intervention_count"] > 0 and metrics["opportunity_realization_rate"] > 0:
        outcome = "P"
    else:
        outcome = "F"
    report = {
        "study": "CONCORDIA v8 FINAL_AUDIT_V8",
        "outcome": outcome,
        "checks": [{"requirement": name, "value": value, "passed": bool(passed)} for name, value, passed in checks],
        "passed_count": sum(bool(passed) for _, _, passed in checks),
        "check_count": len(checks),
        "integrity_passed": bool(
            final["freeze_manifest_unchanged"]
            and osm["frozen_immutable"]
            and final["pairing_failure_count"] == 0
            and final["future_state_leakage_count"] == 0
        ),
        "freeze_manifest_self_hash": manifest["manifest_self_hash"],
        "rl_decision": "NO",
    }
    write_json(ROOT / "artifacts/studies/v8_final_audit/summary.json", report)
    print(json.dumps(report, indent=2))
    if not report["integrity_passed"]:
        raise RuntimeError("v8 final integrity audit failed")


if __name__ == "__main__":
    run()
