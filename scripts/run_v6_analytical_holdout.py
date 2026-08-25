#!/usr/bin/env python3
from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import yaml

from concordia.evaluation import summarize_selective_policy
from concordia.feasibility import build_alignment_case, unified_calibration_metrics
from concordia.selective import V5DecisionInputs
from v5_frozen import analytical_policy, load_deployment, prepare_cases
from v6_frozen import load_policy, verify_frozen, write_json


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/studies/v6_frozen_analytical_holdout"


def _row(case: dict, intervene: bool) -> dict:
    baseline = float(case["baseline_metrics"]["eta_only_ttt"])
    adaptive = float(case["adaptive_counterfactual"]["ttt"])
    success = case["label"] == "WIN"
    return {
        "case_id": case["case_id"],
        "intervene": bool(intervene),
        "success": bool(intervene and success),
        "counterfactual_success": success,
        "system_ttt_gain": baseline - adaptive if intervene else 0.0,
        "relative_ttt_gain": (baseline - adaptive) / max(baseline, 1e-9) if intervene else 0.0,
        "regret_violation": bool(
            intervene and case["adaptive_counterfactual"]["maximum_regret"] > 0.08
        ),
        "safety_violation": bool(
            intervene and case["adaptive_counterfactual"]["safety_difference"] > 0.25
        ),
        "legal_violation": bool(intervene and not case["adaptive_counterfactual"]["legal"]),
    }


def run() -> Path:
    existing = OUTPUT / "summary.json"
    if existing.is_file():
        verify_frozen()
        print(existing)
        return existing
    before = verify_frozen()
    v6_policy = load_policy()
    config = yaml.safe_load((ROOT / "configs/v6/analytical.yaml").read_text())["final_holdout"]
    cases = []
    for scenario in config["scenarios"]:
        for seed in config["seeds"]:
            for demand in config["demand_scale"]:
                for heterogeneity in config["heterogeneity"]:
                    for penetration in config["navigation_penetration"]:
                        cases.append(
                            build_alignment_case(
                                scenario=str(scenario), seed=int(seed), demand_scale=float(demand),
                                heterogeneity=str(heterogeneity),
                                navigation_penetration=float(penetration), user_count=6,
                                regret_limit=0.08,
                                epsilon_grid=[0.0, 0.02, 0.04, 0.08, 0.12, 0.16, 0.24],
                                minimum_relative_ttt_gain=0.01, safety_delta=0.25,
                                source_split="v6_final_analytical_holdout",
                            )
                        )
    if len(cases) != int(config["expected_case_count"]):
        raise RuntimeError("v6 analytical holdout case count mismatch")
    development_seeds = set(
        yaml.safe_load((ROOT / "configs/v6/micro_design.yaml").read_text())["development_seeds"]
    )
    if set(config["seeds"]) & development_seeds:
        raise RuntimeError("v6 analytical final seed leaked into development")
    regime, shift, shift_names, bundle, thresholds = load_deployment()
    started = time.perf_counter()
    rows, prediction = prepare_cases(cases, regime, shift, shift_names, bundle)
    per_case_latency = (time.perf_counter() - started) / len(rows)
    reference = analytical_policy(thresholds, variant="V5-RD")
    reference_mask = []
    stage1_mask = []
    decisions = []
    for index, row in enumerate(rows):
        inputs = V5DecisionInputs(
            row["case_id"], row["regime"], row["shift_class"], row["domain_shift_score"],
            float(prediction.success_probability[index]),
            float(prediction.analytical_benefit[index]),
            float(prediction.corrected_microscopic_benefit[index]),
            float(prediction.microscopic_success_probability[index]),
            float(prediction.microscopic_safety_probability_upper[index]),
            bool(row["adaptive_counterfactual"]["legal"]),
        )
        reference_decision = reference.decide(inputs)
        stage1 = float(prediction.success_probability[index]) >= v6_policy.stage1_threshold
        reference_mask.append(reference_decision.intervene)
        stage1_mask.append(stage1)
        decisions.append(
            {
                "case_id": row["case_id"],
                "v5_rd_reference_intervene": reference_decision.intervene,
                "v6_recall_screen_pass": stage1,
                "analytical_success_probability": float(prediction.success_probability[index]),
            }
        )
    reference_rows = [_row(row, mask) for row, mask in zip(rows, reference_mask)]
    stage1_rows = [_row(row, mask) for row, mask in zip(rows, stage1_mask)]
    reference_metrics = summarize_selective_policy(reference_rows)
    stage1_metrics = summarize_selective_policy(stage1_rows)
    opportunities = sum(row["label"] == "WIN" for row in rows)
    stage1_recall = sum(
        mask and row["label"] == "WIN" for mask, row in zip(stage1_mask, rows)
    ) / max(1, opportunities)
    summary = {
        "complete": True,
        "study": "v6 frozen analytical holdout and recall-oriented stage-1 audit",
        "case_count": len(rows),
        "untouched_before_freeze": True,
        "label_counts": dict(Counter(row["label"] for row in rows)),
        "precision_preserving_reference": "V5-RD",
        "reference_metrics": reference_metrics,
        "reference_precision_target_met": reference_metrics["intervention_precision"] >= 0.80,
        "v6_stage1_metrics_not_a_deployment_claim": stage1_metrics,
        "v6_stage1_opportunity_recall": stage1_recall,
        "v6_stage1_recall_target_met": stage1_recall >= 0.95,
        "calibration": unified_calibration_metrics(
            [int(row["label"] == "WIN") for row in rows], prediction.success_probability
        ),
        "mean_inference_seconds": per_case_latency,
        "freeze_manifest_hash_before": before["manifest_self_hash"],
        "freeze_manifest_hash_after": verify_frozen()["manifest_self_hash"],
        "frozen_immutable": True,
        "rl_used": False,
    }
    write_json(OUTPUT / "raw_metrics.json", rows)
    write_json(OUTPUT / "decisions.json", decisions)
    write_json(OUTPUT / "summary.json", summary)
    print(OUTPUT / "summary.json")
    return OUTPUT / "summary.json"


if __name__ == "__main__":
    run()
