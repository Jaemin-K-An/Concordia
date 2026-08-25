#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from build_final_report_v6 import _outcome
from v5_frozen import verify_frozen as verify_v5
from v6_frozen import verify_frozen as verify_v6, write_json


ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str):
    return json.loads((ROOT / relative).read_text())


def _yes(value: bool) -> str:
    return "YES" if value else "NO"


def run() -> Path:
    v5 = verify_v5()
    before = verify_v6()
    dataset = _read("artifacts/studies/v6_micro_dataset/dataset_summary.json")
    training = _read("artifacts/studies/v6_micro_model_selection/training_manifest.json")
    validation = _read("artifacts/studies/v6_policy_validation/validation_summary.json")
    analytical = _read("artifacts/studies/v6_frozen_analytical_holdout/summary.json")
    micro = _read("artifacts/studies/v6_frozen_micro_holdout/summary.json")
    osm = _read("artifacts/studies/v6_real_topology/summary.json")
    failure = _read("artifacts/studies/v6_failure_analysis/summary.json")
    ablations = _read("artifacts/studies/v6_policy_validation/ablation.json")
    primary = micro["primary_metrics"]
    final_ids = {row["case_id"] for row in _read("artifacts/studies/v6_frozen_micro_holdout/raw_metrics.json")}
    development_ids = {
        case_id for values in training["case_ids"].values() for case_id in values
    }
    ablation_index = {row["ablation"]: row for row in ablations}
    full_candidate = next(
        row
        for row in _read("artifacts/studies/v6_micro_model_selection/candidate_comparison.json")
        if row["name"] == validation["selected"]["model"]
    )
    temporal_useful = full_candidate["validation_metrics"]["average_precision"] > ablation_index[
        "without_traffic_temporal"
    ]["average_precision"]
    analytical_useful = full_candidate["validation_metrics"]["average_precision"] > ablation_index[
        "without_analytical"
    ]["average_precision"]
    tests = subprocess.run(
        ["python3", "-m", "pytest", "tests/test_v6_micro.py", "-q"],
        cwd=ROOT,
        env={"PYTHONPATH": "src:."},
        capture_output=True,
        text=True,
        check=False,
    )
    after = verify_v6()
    outcome = _outcome(micro, osm)
    checks = {
        "new_microscopic_dataset_at_least_500": dataset["pair_count"] >= 500,
        "final_micro_holdout_untouched_and_disjoint": (
            micro["untouched_before_freeze"]
            and not bool(final_ids & development_ids)
            and micro["seed_disjoint_from_development"]
        ),
        "micro_precision_at_least_80_percent": (
            primary["intervention_count"] > 0 and primary["precision"] >= 0.80
        ),
        "micro_coverage_at_least_10_percent": primary["coverage"] >= 0.10,
        "micro_interventions_at_least_30": primary["intervention_count"] >= 30,
        "opportunity_recovery_at_least_40_percent": primary["opportunity_recovery_rate"] >= 0.40,
        "micro_safety_violations_zero": primary["safety_violation_count"] == 0,
        "false_safe_rate_at_most_5_percent": primary["false_safe_rate"] <= 0.05,
        "precision_lower_ci_above_60_percent": primary["precision_wilson_95_lower"] > 0.60,
        "analytical_precision_at_least_80_percent": analytical["reference_precision_target_met"],
        "real_osm_interventions_positive": osm["primary_metrics"]["intervention_count"] > 0,
        "real_osm_safe_success_positive": osm["primary_metrics"]["success_count"] > 0,
        "temporal_features_useful": temporal_useful,
        "analytical_score_useful_in_micro_model": analytical_useful,
        "rl_used": False,
        "v6_unit_tests_pass": tests.returncode == 0,
        "v5_frozen_artifacts_preserved": bool(v5["complete"]),
        "v6_post_freeze_hashes_immutable": before["manifest_self_hash"] == after["manifest_self_hash"],
        "failure_visualization_complete": failure["complete"],
    }
    table = [
        "# FINAL_AUDIT_V6",
        "",
        f"Final Outcome: **{outcome}**",
        "",
        "| Metric | Result | Value |",
        "|---|---:|---:|",
        f"| New microscopic dataset ≥500? | {_yes(checks['new_microscopic_dataset_at_least_500'])} | {dataset['pair_count']} |",
        f"| Final micro holdout untouched? | {_yes(checks['final_micro_holdout_untouched_and_disjoint'])} | {micro['pair_count']} disjoint pairs |",
        f"| Micro precision ≥80%? | {_yes(checks['micro_precision_at_least_80_percent'])} | {primary['precision']:.3f} |",
        f"| Micro coverage ≥10%? | {_yes(checks['micro_coverage_at_least_10_percent'])} | {primary['coverage']:.3f} |",
        f"| Micro interventions ≥30? | {_yes(checks['micro_interventions_at_least_30'])} | {primary['intervention_count']} |",
        f"| Opportunity Recovery ≥40%? | {_yes(checks['opportunity_recovery_at_least_40_percent'])} | {primary['opportunity_recovery_rate']:.3f} |",
        f"| Micro safety violations =0? | {_yes(checks['micro_safety_violations_zero'])} | {primary['safety_violation_count']} |",
        f"| False-safe rate ≤5%? | {_yes(checks['false_safe_rate_at_most_5_percent'])} | {primary['false_safe_rate']:.3f} |",
        f"| Precision lower CI >60%? | {_yes(checks['precision_lower_ci_above_60_percent'])} | {primary['precision_wilson_95_lower']:.3f} |",
        f"| Analytical precision ≥80%? | {_yes(checks['analytical_precision_at_least_80_percent'])} | {analytical['reference_metrics']['intervention_precision']:.3f} |",
        f"| Real OSM interventions >0? | {_yes(checks['real_osm_interventions_positive'])} | {osm['primary_metrics']['intervention_count']} |",
        f"| Real OSM safe success >0? | {_yes(checks['real_osm_safe_success_positive'])} | {osm['primary_metrics']['success_count']} |",
        f"| Temporal features useful? | {_yes(checks['temporal_features_useful'])} | development validation ablation |",
        f"| Analytical score useful in micro model? | {_yes(checks['analytical_score_useful_in_micro_model'])} | development validation ablation |",
        "| RL used? | NO | classification/domain-transfer problem |",
        "",
        "The outcome is computed from frozen final evidence. No final outcome was used for model, calibration, architecture, conformal cutoff, or threshold selection.",
        "",
        "## Reproducibility",
        "",
        f"- v5 frozen source commit preserved: `{v5['source_commit']}`",
        f"- v6 frozen source commit: `{before['source_commit']}`",
        f"- v6 unit tests: `{tests.stdout.strip()}`",
        f"- validation selection: `{validation['selected']['method']}`",
    ]
    path = ROOT / "FINAL_AUDIT_V6.md"
    path.write_text("\n".join(table) + "\n", encoding="utf-8")
    write_json(
        ROOT / "artifacts/v6/final_audit_summary.json",
        {"complete": True, "outcome": outcome, "checks": checks, "unit_test_output": tests.stdout},
    )
    print(path)
    return path


if __name__ == "__main__":
    run()
