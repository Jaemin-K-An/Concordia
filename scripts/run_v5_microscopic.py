#!/usr/bin/env python3
from __future__ import annotations

import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from concordia.evaluation import adaptive_success_claim_allowed, summarize_selective_policy
from concordia.feasibility import build_alignment_case
from concordia.selective import V5DecisionInputs
from run_microscopic_study import _run_one
from build_v5_micro_dataset import _build_network, _compact
from v5_frozen import load_deployment, microscopic_policy, prepare_cases, verify_frozen, write_json


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/studies/v5_microscopic_holdout"


def _summary(rows: list[dict]) -> dict:
    metrics = summarize_selective_policy(rows)
    metrics.update(
        {
            "safe_success_count": sum(
                row["intervene"] and row["success"] and not row["safety_violation"]
                for row in rows
            ),
            "false_safe_rate": metrics["safety_violation_count"]
            / max(1, metrics["intervention_count"]),
            "mean_microscopic_relative_ttt_gain": float(
                np.mean(
                    [row["microscopic_benefit"] for row in rows if row["intervene"]]
                )
            )
            if any(row["intervene"] for row in rows)
            else 0.0,
        }
    )
    return metrics


def run() -> Path:
    existing = OUTPUT / "summary.json"
    if existing.is_file():
        verify_frozen()
        print(existing)
        return existing
    verify_frozen()
    config = yaml.safe_load((ROOT / "configs/v5/microscopic.yaml").read_text())
    base = yaml.safe_load(
        (ROOT / "configs/experiments/microscopic_policy_matrix.yaml").read_text()
    )
    base.update(
        {
            "vehicle_generation_seconds": config["vehicle_generation_seconds"],
            "maximum_simulation_seconds": config["maximum_simulation_seconds"],
            "preference_epsilon": config["preference_epsilon"],
        }
    )
    conditions = [
        (int(seed), str(t[0]), int(t[1]), float(t[2]), str(t[3]))
        for seed in config["final_holdout_seeds"]
        for t in config["condition_templates"]
    ]
    cases = []
    for seed, scenario, demand, penetration, heterogeneity in conditions:
        case = build_alignment_case(
            scenario=scenario,
            seed=seed,
            demand_scale=demand / 1200.0,
            heterogeneity=heterogeneity,
            navigation_penetration=penetration,
            user_count=6,
            regret_limit=0.08,
            epsilon_grid=[0.0, 0.02, 0.04, 0.08, 0.12, 0.16],
            minimum_relative_ttt_gain=0.01,
            safety_delta=0.25,
            source_split="v5_microscopic_final_pre_run",
        )
        case["micro_condition"] = {
            "demand_vehicles_per_hour": demand,
            "heterogeneity": heterogeneity,
            "navigation_penetration": penetration,
        }
        cases.append(case)
    regime, shift, shift_names, bundle, thresholds = load_deployment()
    prepared, prediction = prepare_cases(cases, regime, shift, shift_names, bundle)
    policy = microscopic_policy(thresholds)
    decisions = {}
    for index, row in enumerate(prepared):
        decision = policy.decide(
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
        key = conditions[index]
        decisions[key] = {
            "intervene": decision.intervene,
            "reasons": list(decision.reasons),
            "explanation": list(decision.explanation),
            "analytical_probability": float(prediction.success_probability[index]),
            "analytical_benefit": float(prediction.analytical_benefit[index]),
            "corrected_microscopic_benefit": float(
                prediction.corrected_microscopic_benefit[index]
            ),
            "microscopic_success_probability": float(
                prediction.microscopic_success_probability[index]
            ),
            "microscopic_safety_probability_upper": float(
                prediction.microscopic_safety_probability_upper[index]
            ),
            "regime": row["regime"],
            "shift_class": row["shift_class"],
            "domain_shift_score": row["domain_shift_score"],
        }
    with tempfile.TemporaryDirectory(prefix="concordia-v5-micro-final-") as temporary:
        directory = Path(temporary)
        networks = {
            scenario: _build_network(directory, scenario)
            for scenario in sorted({condition[1] for condition in conditions})
        }
        actual = []
        for seed, scenario, demand, penetration, heterogeneity in conditions:
            for baseline_policy in ("B1", "B6"):
                row = _run_one(
                    networks[scenario],
                    base,
                    baseline_policy,
                    seed,
                    demand,
                    penetration,
                    heterogeneity,
                )
                row["scenario"] = scenario
                actual.append(_compact(row))
    actual_index = {}
    for row in actual:
        key = (
            row["seed"],
            row["scenario"],
            row["demand_vehicles_per_hour"],
            row["navigation_penetration"],
            row["heterogeneity"],
        )
        actual_index.setdefault(key, {})[row["policy"]] = row
    policy_rows = []
    b6_rows = []
    pair_rows = []
    for key in conditions:
        b1, b6 = actual_index[key]["B1"], actual_index[key]["B6"]
        gain = (b1["total_travel_time_seconds"] - b6["total_travel_time_seconds"]) / max(
            b1["total_travel_time_seconds"], 1e-9
        )
        safety_difference = b6["safety"]["cvar_drac_95"] - b1["safety"]["cvar_drac_95"]
        unsafe = safety_difference > 0.25
        success = gain >= 0.01 and not unsafe
        decision = decisions[key]
        failure_type = (
            "none"
            if success
            else "microscopic_safety_mismatch"
            if unsafe
            else "partial_adoption_feedback"
            if b6["acceptance_rate"] < 0.5
            else "benefit_prediction_error"
        )
        common = {
            "case_id": f"v5-micro-final-{key[1]}-s{key[0]}-d{key[2]}-p{key[3]:.2f}-{key[4]}",
            "counterfactual_success": success,
            "system_ttt_gain": b1["total_travel_time_seconds"] - b6["total_travel_time_seconds"],
            "microscopic_benefit": gain,
            "regret_violation": False,
            "safety_violation": unsafe,
            "legal_violation": False,
        }
        policy_rows.append(
            {
                **common,
                "intervene": decision["intervene"],
                "success": bool(decision["intervene"] and success),
            }
        )
        b6_rows.append({**common, "intervene": True, "success": success})
        pair_rows.append(
            {
                **common,
                **decision,
                "seed": key[0],
                "scenario": key[1],
                "demand_vehicles_per_hour": key[2],
                "navigation_penetration": key[3],
                "heterogeneity": key[4],
                "failure_type": failure_type,
                "realized_acceptance": b6["acceptance_rate"],
                "microscopic_safety_difference": safety_difference,
                "b1": b1,
                "b6": b6,
            }
        )
    if len(pair_rows) != int(config["final_holdout_pair_count"]):
        raise RuntimeError("v5 microscopic final pair count mismatch")
    metrics = _summary(policy_rows)
    baseline_metrics = _summary(b6_rows)
    claim_allowed = adaptive_success_claim_allowed(
        metrics, int(config["minimum_claim_interventions"])
    )
    hypotheses = {
        "H24_micro_correction_reduces_benefit_mae": {
            "analytical_mae": float(
                np.mean(
                    [
                        abs(row["microscopic_benefit"] - row["analytical_benefit"])
                        for row in pair_rows
                    ]
                )
            ),
            "corrected_mae": float(
                np.mean(
                    [
                        abs(
                            row["microscopic_benefit"]
                            - row["corrected_microscopic_benefit"]
                        )
                        for row in pair_rows
                    ]
                )
            ),
        },
        "H25_zero_safety_violations": metrics["safety_violation_count"] == 0,
        "H25_false_safe_target": metrics["false_safe_rate"] <= 0.05,
    }
    summary = {
        "complete": True,
        "study": "v5 frozen microscopic domain bridge",
        "actual_sumo": True,
        "untouched_holdout": True,
        "pair_count": len(pair_rows),
        "selected_policy": "V5-F-microscopic",
        "primary_metrics": metrics,
        "always_adapt_metrics": baseline_metrics,
        "claim_eligible": claim_allowed,
        "failure_taxonomy": dict(Counter(row["failure_type"] for row in pair_rows)),
        "hypotheses": hypotheses,
        "final_seed_disjoint_from_development": not bool(
            set(config["final_holdout_seeds"]) & set(config["development_seeds"])
        ),
        "rl_used": False,
        "claim_boundary": config["claim_boundary"],
    }
    write_json(OUTPUT / "raw_metrics.json", pair_rows)
    write_json(OUTPUT / "policy_rows.json", policy_rows)
    write_json(OUTPUT / "summary.json", summary)
    print(OUTPUT / "summary.json")
    return OUTPUT / "summary.json"


if __name__ == "__main__":
    run()
