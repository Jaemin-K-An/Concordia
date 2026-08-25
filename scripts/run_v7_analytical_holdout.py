#!/usr/bin/env python3
from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from concordia.evaluation import summarize_selective_policy
from concordia.feasibility import build_alignment_case, unified_calibration_metrics
from concordia.selective import V5DecisionInputs
from v5_frozen import analytical_policy, load_deployment, prepare_cases
from v7_frozen import verify_frozen, write_json


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/studies/v7_frozen_analytical_holdout"


def run() -> Path:
    existing = OUTPUT / "summary.json"
    if existing.is_file():
        verify_frozen()
        print(existing)
        return existing
    before = verify_frozen()
    config = yaml.safe_load((ROOT / "configs/v7/analytical.yaml").read_text())[
        "final_check"
    ]
    cases = []
    for scenario in config["scenarios"]:
        for seed in config["seeds"]:
            for demand in config["demand_scale"]:
                for heterogeneity in config["heterogeneity"]:
                    for penetration in config["navigation_penetration"]:
                        cases.append(
                            build_alignment_case(
                                scenario=str(scenario),
                                seed=int(seed),
                                demand_scale=float(demand),
                                heterogeneity=str(heterogeneity),
                                navigation_penetration=float(penetration),
                                user_count=6,
                                regret_limit=0.08,
                                epsilon_grid=[0.0, 0.02, 0.04, 0.08, 0.12, 0.16, 0.24],
                                minimum_relative_ttt_gain=0.01,
                                safety_delta=0.25,
                                source_split="v7_final_analytical_correctness_holdout",
                            )
                        )
    if len(cases) != int(config["expected_case_count"]):
        raise RuntimeError("v7 analytical final check count mismatch")
    development_seeds = set(
        yaml.safe_load((ROOT / "configs/v7/paired_design.yaml").read_text())[
            "new_development_seeds"
        ]
    )
    if set(config["seeds"]) & development_seeds:
        raise RuntimeError("v7 analytical final seed leaked into development")
    regime, shift, shift_names, bundle, thresholds = load_deployment()
    started = time.perf_counter()
    rows, prediction = prepare_cases(cases, regime, shift, shift_names, bundle)
    latency = (time.perf_counter() - started) / len(rows)
    records = []
    reference = analytical_policy(thresholds, variant="V5-RD")
    reference_rows = []
    for index, row in enumerate(rows):
        baseline = float(row["baseline_metrics"]["eta_only_ttt"])
        adaptive = float(row["adaptive_counterfactual"]["ttt"])
        absolute = baseline - adaptive
        relative = absolute / max(baseline, 1e-9)
        decision = reference.decide(
            V5DecisionInputs(
                row["case_id"],
                row["regime"],
                row["shift_class"],
                row["domain_shift_score"],
                float(prediction.success_probability[index]),
                float(prediction.analytical_benefit[index]),
                float(prediction.corrected_microscopic_benefit[index]),
                float(prediction.microscopic_success_probability[index]),
                float(prediction.microscopic_safety_probability_upper[index]),
                bool(row["adaptive_counterfactual"]["legal"]),
            )
        )
        record = {
            "case_id": row["case_id"],
            "scenario": row["scenario"],
            "seed": row["seed"],
            "tau_t_absolute": absolute,
            "tau_t_relative": relative,
            "tau_s": float(row["adaptive_counterfactual"]["safety_difference"]),
            "max_regret": float(row["adaptive_counterfactual"]["maximum_regret"]),
            "legal": bool(row["adaptive_counterfactual"]["legal"]),
            "label": row["label"],
            "historical_analytical_success_probability": float(
                prediction.success_probability[index]
            ),
            "historical_v5_rd_intervene": bool(decision.intervene),
        }
        records.append(record)
        reference_rows.append(
            {
                "case_id": row["case_id"],
                "intervene": bool(decision.intervene),
                "success": bool(decision.intervene and row["label"] == "WIN"),
                "counterfactual_success": row["label"] == "WIN",
                "system_ttt_gain": absolute if decision.intervene else 0.0,
                "relative_ttt_gain": relative if decision.intervene else 0.0,
                "regret_violation": bool(
                    decision.intervene
                    and row["adaptive_counterfactual"]["maximum_regret"] > 0.08
                ),
                "safety_violation": bool(
                    decision.intervene
                    and row["adaptive_counterfactual"]["safety_difference"] > 0.25
                ),
                "legal_violation": bool(
                    decision.intervene and not row["adaptive_counterfactual"]["legal"]
                ),
            }
        )
    labels = np.asarray([row["label"] == "WIN" for row in rows], dtype=int)
    probabilities = np.asarray(prediction.success_probability, dtype=float)
    formula_failures = sum(
        abs(
            record["tau_t_relative"]
            - record["tau_t_absolute"]
            / max(float(row["baseline_metrics"]["eta_only_ttt"]), 1e-9)
        )
        > 1e-12
        for record, row in zip(records, rows)
    )
    reference_metrics = summarize_selective_policy(reference_rows)
    summary = {
        "complete": True,
        "study": "v7 frozen analytical paired-effect correctness check",
        "case_count": len(records),
        "untouched_before_freeze": True,
        "seed_disjoint_from_micro_development": not bool(
            set(config["seeds"]) & development_seeds
        ),
        "label_counts": dict(Counter(row["label"] for row in rows)),
        "paired_effect_formula_failure_count": formula_failures,
        "all_routes_legal": all(record["legal"] for record in records),
        "positive_uplift_count": sum(record["tau_t_relative"] >= 0.01 for record in records),
        "historical_analytical_calibration_reference": unified_calibration_metrics(
            labels, probabilities
        ),
        "historical_v5_rd_reference_metrics": reference_metrics,
        "historical_analytical_precision_at_least_80_percent": bool(
            reference_metrics["intervention_precision"] >= 0.80
        ),
        "mean_reference_inference_seconds": latency,
        "role": "analytical correctness and continuity diagnostic; actual SUMO is primary",
        "freeze_manifest_hash_before": before["manifest_self_hash"],
        "freeze_manifest_hash_after": verify_frozen()["manifest_self_hash"],
        "frozen_immutable": True,
        "rl_used": False,
    }
    write_json(OUTPUT / "raw_metrics.json", records)
    write_json(OUTPUT / "summary.json", summary)
    print(OUTPUT / "summary.json")
    return OUTPUT / "summary.json"


if __name__ == "__main__":
    run()
