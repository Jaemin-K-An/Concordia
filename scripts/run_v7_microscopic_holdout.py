#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml

from build_v7_paired_dataset import _analytical_predictions, _execute_task, _v6_scores
from concordia.feasibility import build_alignment_case
from concordia.micro_v6.features import MICRO_V6_FEATURE_SCHEMA
from concordia.uplift_v7.evaluation import (
    cumulative_gain,
    deployment_metrics,
    effect_calibration,
    regression_metrics,
)
from concordia.uplift_v7.paired_dataset import feature_matrix, paired_row_from_v6
from concordia.uplift_v7.policy import UpliftPolicy
from concordia.selective import V5DecisionInputs
from v5_frozen import load_deployment, microscopic_policy, prepare_cases
from v6_frozen import load_policy as load_v6_policy
from v7_frozen import load_policy, verify_frozen, write_json
from v7_micro_sim import build_v7_network


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/studies/v7_frozen_micro_holdout"
CHECKPOINT = OUTPUT / "final_checkpoint.json"


def _variant_mask(policy: UpliftPolicy, rows: list[dict], method: str) -> list[bool]:
    matrix = feature_matrix(rows, policy.traffic_model.feature_names)
    traffic = policy.traffic_model.predict(matrix)
    safety = policy.safety_model.predict(matrix)
    regret = policy.regret_model.predict(matrix)
    if method == "mean":
        traffic_lower, safety_upper, regret_upper = traffic, safety, regret
    elif method == "bootstrap_quantile":
        _mean, traffic_lower, _upper = policy.traffic_bootstrap.interval(matrix, 0.10, 0.90)
        _mean, _lower, safety_upper = policy.safety_bootstrap.interval(matrix, 0.10, 0.90)
        _mean, _lower, regret_upper = policy.regret_bootstrap.interval(matrix, 0.10, 0.90)
    else:
        traffic_lower = traffic - policy.conformal.traffic_radius
        safety_upper = safety + policy.conformal.safety_upper_adjustment
        regret_upper = regret + policy.conformal.regret_upper_adjustment
    return [
        bool(
            traffic_lower[index] > 0.01
            and safety_upper[index] <= 0.25
            and regret_upper[index] <= 0.08
            and row["outcomes"]["legal"]
        )
        for index, row in enumerate(rows)
    ]


def _v6_mask(rows: list[dict]) -> list[bool]:
    converted = [
        {
            "case_id": row["pair_id"],
            "features_pre_decision": {
                name: row["predecision_features"][name]
                for name in MICRO_V6_FEATURE_SCHEMA
            },
            "label": {"safe_micro_success": row["outcomes"]["safe_micro_success"]},
        }
        for row in rows
    ]
    return [decision["intervene"] for decision in load_v6_policy().decide(converted)]


def _v5_mask(tasks: list[tuple[int, dict]]) -> list[bool]:
    cases = []
    for seed, condition in tasks:
        topology = str(condition["topology"])
        scenario = "two_route" if topology in {"real_like", "asymmetric"} else topology
        cases.append(
            build_alignment_case(
                scenario=scenario,
                seed=seed,
                demand_scale=float(condition["demand"]) / 1200.0,
                heterogeneity=str(condition["heterogeneity"]),
                navigation_penetration=float(condition["penetration"]),
                user_count=6,
                regret_limit=0.08,
                epsilon_grid=[0.0, 0.02, 0.04, 0.08, 0.12, 0.16],
                minimum_relative_ttt_gain=0.01,
                safety_delta=0.25,
                source_split="v7_final_v5_fair_comparator",
            )
        )
    regime, shift, shift_names, bundle, thresholds = load_deployment()
    prepared, prediction = prepare_cases(cases, regime, shift, shift_names, bundle)
    policy = microscopic_policy(thresholds)
    output = []
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
        output.append(bool(decision.intervene))
    return output


def _component_mask(
    policy: UpliftPolicy,
    rows: list[dict],
    *,
    safety: bool,
    regret: bool,
) -> list[bool]:
    bounds = policy.predict_bounds(rows)
    return [
        bool(
            bounds["traffic_lower"][index] > policy.traffic_lcb_threshold
            and (
                not safety
                or bounds["safety_upper"][index] <= policy.safety_ucb_threshold
            )
            and (
                not regret
                or bounds["regret_upper"][index] <= policy.regret_ucb_threshold
            )
            and row["outcomes"]["legal"]
        )
        for index, row in enumerate(rows)
    ]


def _population_effect(rows: list[dict], mask: list[bool]) -> dict:
    baseline = sum(float(row["ttt_b1"]) for row in rows)
    executed = sum(
        float(row["ttt_adaptive"] if selected else row["ttt_b1"])
        for row, selected in zip(rows, mask)
    )
    return {
        "baseline_population_ttt_seconds": baseline,
        "executed_population_ttt_seconds": executed,
        "population_ttt_gain_seconds": baseline - executed,
        "population_relative_ttt_gain": (baseline - executed) / max(baseline, 1e-9),
    }


def _error_analysis(rows: list[dict], mask: list[bool], predictions: np.ndarray) -> dict:
    false_positive = []
    false_negative = []
    grouped = defaultdict(lambda: {"count": 0, "selected": 0, "success": 0})
    for row, selected, prediction in zip(rows, mask, predictions):
        success = bool(row["outcomes"]["safe_micro_success"])
        key = (
            row["condition"]["topology"],
            row["condition"]["perturbation"],
            row["outcomes"]["benefit_magnitude_bin"],
        )
        grouped[key]["count"] += 1
        grouped[key]["selected"] += int(selected)
        grouped[key]["success"] += int(success)
        payload = {
            "pair_id": row["pair_id"],
            "predicted_relative_uplift": float(prediction),
            "realized_relative_uplift": float(row["outcomes"]["tau_t_relative"]),
            "safety_effect": float(row["outcomes"]["tau_s"]),
            "regret": float(row["outcomes"]["max_regret"]),
            "topology": row["condition"]["topology"],
        }
        if selected and not success:
            false_positive.append(payload)
        elif not selected and success:
            false_negative.append(payload)
    return {
        "false_positive_count": len(false_positive),
        "false_negative_count": len(false_negative),
        "false_positives": false_positive,
        "false_negatives": sorted(
            false_negative,
            key=lambda row: -row["realized_relative_uplift"],
        )[:50],
        "scenario_cells": [
            {
                "topology": key[0],
                "perturbation": key[1],
                "benefit_magnitude_bin": key[2],
                **value,
            }
            for key, value in sorted(grouped.items())
        ],
    }


def run(*, workers: int = 4) -> Path:
    existing = OUTPUT / "summary.json"
    if existing.is_file():
        verify_frozen()
        print(existing)
        return existing
    before = verify_frozen()
    policy = load_policy()
    design = yaml.safe_load((ROOT / "configs/v7/paired_design.yaml").read_text())
    development_seeds = set(design["new_development_seeds"])
    development_seeds.update(
        int(row["seed"])
        for row in json.loads(
            (ROOT / "artifacts/studies/v7_paired_dataset/raw_metrics.json").read_text()
        )
    )
    final_seeds = set(map(int, design["final_holdout_seeds"]))
    if development_seeds & final_seeds:
        raise RuntimeError("v7 microscopic final seed leaked into development")
    tasks = [
        (int(seed), dict(condition))
        for seed in design["final_holdout_seeds"]
        for condition in design["condition_templates"]
    ]
    if len(tasks) != int(design["final_holdout_pair_count"]):
        raise RuntimeError("v7 final microscopic pair count mismatch")
    analytical = _analytical_predictions(tasks)
    completed = {
        row["case_id"]: row
        for row in (json.loads(CHECKPOINT.read_text()) if CHECKPOINT.is_file() else [])
    }
    with tempfile.TemporaryDirectory(prefix="concordia-v7-final-") as temporary:
        directory = Path(temporary)
        networks = {}
        for condition in design["condition_templates"]:
            key = (condition["topology"], condition["perturbation"])
            if key not in networks:
                networks[key] = build_v7_network(directory, *key)
        pending = []
        for seed, condition in tasks:
            case_id = f"v7-development-{condition['id']}-s{seed}"
            if case_id in completed:
                continue
            network, metadata = networks[(condition["topology"], condition["perturbation"])]
            pending.append(
                {
                    "network": str(network),
                    "metadata": metadata,
                    "config": design,
                    "condition": condition,
                    "seed": seed,
                    "analytical": analytical[f"{condition['id']}::{seed}"],
                }
            )
        if pending:
            with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
                futures = {executor.submit(_execute_task, task): task for task in pending}
                for future in as_completed(futures):
                    row = future.result()
                    completed[row["case_id"]] = row
                    if len(completed) % 10 == 0:
                        write_json(
                            CHECKPOINT,
                            sorted(completed.values(), key=lambda value: value["case_id"]),
                        )
    raw = sorted(completed.values(), key=lambda value: value["case_id"])
    if len(raw) != len(tasks):
        raise RuntimeError("v7 final microscopic dataset is incomplete")
    scores = _v6_scores(raw)
    rows = []
    for raw_row, score in zip(raw, scores):
        row = paired_row_from_v6(
            raw_row,
            v6_micro_success_score=score,
            source="v7_untouched_final_actual_sumo",
        )
        row["pair_id"] = raw_row["case_id"].replace("development", "final")
        rows.append(row)
    final_ids = {row["pair_id"] for row in rows}
    training_manifest = json.loads(
        (ROOT / "artifacts/studies/v7_model_selection/training_manifest.json").read_text()
    )
    development_ids = {
        case_id for values in training_manifest["case_ids"].values() for case_id in values
    }
    started = time.perf_counter()
    decisions = policy.decide(rows)
    per_case_latency = (time.perf_counter() - started) / len(rows)
    masks = {
        "B1": [False] * len(rows),
        "B6": [True] * len(rows),
        "V5-F": _v5_mask(tasks),
        "V6-Binary": _v6_mask(rows),
        "V7_mean": _variant_mask(policy, rows, "mean"),
        "V7_quantile": _variant_mask(policy, rows, "bootstrap_quantile"),
        "V7_conformal": _variant_mask(policy, rows, "conformalized_residual"),
        "V7-T": _component_mask(policy, rows, safety=False, regret=False),
        "V7-TS": _component_mask(policy, rows, safety=True, regret=False),
        "V7-TSU": _component_mask(policy, rows, safety=True, regret=True),
        "V7-F": [decision["intervene"] for decision in decisions],
    }
    metrics = {
        name: {**deployment_metrics(rows, mask), **_population_effect(rows, mask)}
        for name, mask in masks.items()
    }
    primary = metrics["V7-F"]
    if primary["safety_violation_count"] > 0 or primary["deployment_precision"] < 0.60:
        outcome = "F"
    elif (
        primary["deployment_precision"] >= 0.80
        and primary["coverage"] >= 0.10
        and primary["intervention_count"] >= 30
        and primary["opportunity_recovery_rate"] >= 0.40
    ):
        outcome = "S"
    else:
        outcome = "P"
    matrix = feature_matrix(rows, policy.traffic_model.feature_names)
    prediction = policy.traffic_model.predict(matrix)
    safety_prediction = policy.safety_model.predict(matrix)
    actual = np.asarray([row["outcomes"]["tau_t_relative"] for row in rows])
    safety_actual = np.asarray([row["outcomes"]["tau_s"] for row in rows])
    safety_bounds = policy.predict_bounds(rows)["safety_upper"]
    predicted_safe = safety_bounds <= policy.safety_ucb_threshold
    primary_mask = masks["V7-F"]
    summary = {
        "complete": True,
        "study": "v7 frozen actual-SUMO paired-treatment-effect final holdout",
        "untouched_before_freeze": True,
        "actual_sumo": True,
        "pair_count": len(rows),
        "actual_sumo_run_count": 2 * len(rows),
        "outcome_counts": dict(
            Counter(row["outcomes"]["benefit_magnitude_bin"] for row in rows)
        ),
        "safe_micro_success_count": sum(
            row["outcomes"]["safe_micro_success"] for row in rows
        ),
        "policy_metrics": metrics,
        "primary_metrics": primary,
        "provisional_outcome": outcome,
        "traffic_effect_metrics": regression_metrics(actual, prediction),
        "safety_effect_metrics": regression_metrics(safety_actual, safety_prediction),
        "safety_effect_false_safe_rate": float(
            np.mean(predicted_safe & (safety_actual > 0.25))
        ),
        "effect_calibration": effect_calibration(actual, prediction),
        "cumulative_gain": cumulative_gain(actual, prediction),
        "seed_disjoint_from_development": not bool(final_seeds & development_seeds),
        "case_ids_absent_from_model_selection": not bool(final_ids & development_ids),
        "pairing_failure_count": sum(
            not row["pairing"]["metadata_identical_except_treatment"] for row in rows
        ),
        "mean_predictor_inference_seconds": per_case_latency,
        "freeze_manifest_hash_before": before["manifest_self_hash"],
        "freeze_manifest_hash_after": verify_frozen()["manifest_self_hash"],
        "frozen_immutable": True,
        "rl_used": False,
    }
    write_json(OUTPUT / "raw_metrics.json", rows)
    write_json(OUTPUT / "decision_log.json", decisions)
    write_json(OUTPUT / "policy_masks.json", masks)
    write_json(OUTPUT / "error_analysis.json", _error_analysis(rows, primary_mask, prediction))
    write_json(OUTPUT / "summary.json", summary)
    CHECKPOINT.unlink(missing_ok=True)
    print(OUTPUT / "summary.json")
    return OUTPUT / "summary.json"


if __name__ == "__main__":
    run()
