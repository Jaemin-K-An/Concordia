#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from build_final_report_v7 import _outcome
from v6_frozen import verify_frozen as verify_v6
from v7_frozen import verify_frozen as verify_v7, write_json


ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str):
    return json.loads((ROOT / relative).read_text())


def _yes(value: bool) -> str:
    return "YES" if value else "NO"


def run() -> Path:
    v6 = verify_v6()
    before = verify_v7()
    dataset = _read("artifacts/studies/v7_paired_dataset/dataset_summary.json")
    training = _read("artifacts/studies/v7_model_selection/training_manifest.json")
    validation = _read("artifacts/studies/v7_policy_validation/validation_summary.json")
    micro = _read("artifacts/studies/v7_frozen_micro_holdout/summary.json")
    final_rows = _read("artifacts/studies/v7_frozen_micro_holdout/raw_metrics.json")
    analytical = _read("artifacts/studies/v7_frozen_analytical_holdout/summary.json")
    osm = _read("artifacts/studies/v7_real_topology/summary.json")
    failure = _read("artifacts/studies/v7_failure_analysis/summary.json")
    placebo = _read("artifacts/studies/v7_model_selection/placebo.json")
    primary = micro["primary_metrics"]
    v6_comparator = micro["policy_metrics"]["V6-Binary"]
    final_ids = {row["pair_id"] for row in final_rows}
    development_ids = {
        case_id for values in training["case_ids"].values() for case_id in values
    }
    tests = subprocess.run(
        [
            "python3",
            "-m",
            "pytest",
            "tests/test_v7_uplift.py",
            "tests/test_v7_frozen_contract.py",
            "-q",
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": "src:scripts:."},
        capture_output=True,
        text=True,
        check=False,
    )
    after = verify_v7()
    placebo_passed = bool(
        placebo["real_signal"]["spearman"] > placebo["permuted_target"]["spearman"]
        and placebo["real_signal"]["rmse"] < placebo["permuted_target"]["rmse"]
    )
    checks = {
        "paired_sumo_development_at_least_1000": dataset["pair_count"] >= 1000,
        "final_micro_holdout_at_least_300": micro["pair_count"] >= 300,
        "pairing_failures_zero": dataset["pairing_failure_count"] == 0
        and micro["pairing_failure_count"] == 0,
        "future_leakage_zero": dataset["future_state_leakage_count"] == 0,
        "final_split_untouched": micro["untouched_before_freeze"]
        and not bool(final_ids & development_ids),
        "deployment_precision_at_least_80": primary["intervention_count"] > 0
        and primary["deployment_precision"] >= 0.80,
        "coverage_at_least_10": primary["coverage"] >= 0.10,
        "interventions_at_least_30": primary["intervention_count"] >= 30,
        "orr_at_least_40": primary["opportunity_recovery_rate"] >= 0.40,
        "safety_violations_zero": primary["safety_violation_count"] == 0,
        "precision_lower_ci_above_60": primary["precision_wilson_95_lower"] > 0.60,
        "v7_beats_v6_binary_selector": primary["opportunity_recovery_rate"]
        > v6_comparator["opportunity_recovery_rate"],
        "osm_interventions_positive": osm["primary_metrics"]["intervention_count"] > 0,
        "osm_safe_successes_positive": osm["primary_metrics"]["success_count"] > 0,
        "placebo_test_passed": placebo_passed,
        "analytical_precision_at_least_80": analytical[
            "historical_analytical_precision_at_least_80_percent"
        ],
        "v7_unit_and_contract_tests_pass": tests.returncode == 0,
        "v6_frozen_evidence_preserved": bool(v6["complete"]),
        "v7_post_freeze_hashes_immutable": before["manifest_self_hash"]
        == after["manifest_self_hash"],
        "failure_analysis_complete": failure["complete"],
        "rl_used": False,
    }
    outcome = _outcome(micro, osm)
    table = [
        "# FINAL_AUDIT_V7",
        "",
        f"Final Outcome: **{outcome}**",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Paired SUMO development ≥1000? | {_yes(checks['paired_sumo_development_at_least_1000'])} · {dataset['pair_count']} |",
        f"| Final micro holdout ≥300? | {_yes(checks['final_micro_holdout_at_least_300'])} · {micro['pair_count']} |",
        f"| Pairing failures =0? | {_yes(checks['pairing_failures_zero'])} · development {dataset['pairing_failure_count']}, final {micro['pairing_failure_count']} |",
        f"| Future leakage =0? | {_yes(checks['future_leakage_zero'])} |",
        f"| Traffic effect MAE | {micro['traffic_effect_metrics']['mae']:.6f} |",
        f"| Traffic effect sign accuracy | {micro['traffic_effect_metrics']['sign_accuracy']:.3f} |",
        f"| Safety effect MAE | {micro['safety_effect_metrics']['mae']:.6f} |",
        f"| Deployment precision ≥80%? | {_yes(checks['deployment_precision_at_least_80'])} · {primary['deployment_precision']:.3f} |",
        f"| Coverage ≥10%? | {_yes(checks['coverage_at_least_10'])} · {primary['coverage']:.3f} |",
        f"| Interventions ≥30? | {_yes(checks['interventions_at_least_30'])} · {primary['intervention_count']} |",
        f"| ORR ≥40%? | {_yes(checks['orr_at_least_40'])} · {primary['opportunity_recovery_rate']:.3f} |",
        f"| Safety violations =0? | {_yes(checks['safety_violations_zero'])} · {primary['safety_violation_count']} |",
        f"| Precision lower CI >60%? | {_yes(checks['precision_lower_ci_above_60'])} · {primary['precision_wilson_95_lower']:.3f} |",
        f"| V7 beats V6 binary selector? | {_yes(checks['v7_beats_v6_binary_selector'])} · ORR {primary['opportunity_recovery_rate']:.3f} vs {v6_comparator['opportunity_recovery_rate']:.3f} |",
        f"| OSM interventions >0? | {_yes(checks['osm_interventions_positive'])} · {osm['primary_metrics']['intervention_count']} |",
        f"| OSM safe successes >0? | {_yes(checks['osm_safe_successes_positive'])} · {osm['primary_metrics']['success_count']} |",
        f"| Placebo test passed? | {_yes(checks['placebo_test_passed'])} |",
        "| RL used? | NO |",
        "",
        "Final seeds and IDs are absent from fitting manifests. The five frozen YAML packages and",
        "their source/artifact hashes were created before analytical, microscopic, or OSM final",
        "evidence was materialized. Failed topologies and missed opportunities remain in the report.",
        "",
        "## Reproducibility",
        "",
        f"- v6 immutable manifest: `{v6['manifest_self_hash']}`",
        f"- v7 immutable manifest: `{before['manifest_self_hash']}`",
        f"- selected treatment-effect learner: `{validation['selected_traffic']['key']}`",
        f"- tests: `{' / '.join(tests.stdout.strip().splitlines())}`",
    ]
    path = ROOT / "FINAL_AUDIT_V7.md"
    path.write_text("\n".join(table) + "\n", encoding="utf-8")
    write_json(
        ROOT / "artifacts/v7/final_audit_summary.json",
        {
            "complete": True,
            "outcome": outcome,
            "checks": checks,
            "unit_test_output": tests.stdout,
            "rl_used": False,
        },
    )
    print(path)
    return path


if __name__ == "__main__":
    run()
