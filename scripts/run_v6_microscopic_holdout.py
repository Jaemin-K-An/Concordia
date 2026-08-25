#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml

from build_v6_micro_dataset import _analytical_predictions, _compact
from concordia.feasibility import build_alignment_case
from concordia.micro_v6 import build_safe_micro_label, claim_allowed, selective_metrics
from concordia.selective import V5DecisionInputs
from v5_frozen import load_deployment, microscopic_policy, prepare_cases
from v6_frozen import load_policy, verify_frozen, write_json
from v6_micro_sim import build_v6_network, run_v6_pair


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/studies/v6_frozen_micro_holdout"
CHECKPOINT = OUTPUT / "final_checkpoint.json"


def _execute(task: dict) -> dict:
    baseline, adaptive = run_v6_pair(
        Path(task["network"]),
        task["metadata"],
        task["config"],
        task["condition"],
        int(task["seed"]),
        task["analytical"],
    )
    label_config = task["preregistration"]["label"]
    label = build_safe_micro_label(
        baseline,
        adaptive,
        minimum_relative_ttt_gain=float(label_config["minimum_relative_ttt_gain"]),
        safety_margin=float(label_config["safety_cvar_drac_margin"]),
        regret_limit=float(label_config["regret_limit"]),
    )
    condition = task["condition"]
    return {
        "case_id": f"v6-final-{condition['id']}-s{task['seed']}",
        "seed": int(task["seed"]),
        "template_id": condition["id"],
        "development_role": "final_holdout",
        "condition": condition,
        "decision_time": baseline["decision_time"],
        "feature_observation_end_time": baseline["feature_observation_end_time"],
        "features_pre_decision": baseline["features_pre_decision"],
        "predecision_series": baseline["predecision_series"],
        "analytical_screening": task["analytical"],
        "counterfactual_B1": _compact(baseline),
        "counterfactual_adaptive": _compact(adaptive),
        "label": label.to_dict(),
        "pairing": {
            "same_seed": baseline["seed"] == adaptive["seed"],
            "same_network_hash": baseline["network_hash"] == adaptive["network_hash"],
            "same_route_file_hash": baseline["route_file_hash"] == adaptive["route_file_hash"],
            "common_random_numbers": True,
        },
    }


def _v5_decisions(tasks: list[tuple[int, dict]]) -> dict[str, bool]:
    cases = []
    for seed, condition in tasks:
        scenario = "two_route" if condition["topology"] == "real_like" else condition["topology"]
        case = build_alignment_case(
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
            source_split="v6_final_v5_historical_baseline_predecision",
        )
        cases.append(case)
    regime, shift, shift_names, bundle, thresholds = load_deployment()
    rows, prediction = prepare_cases(cases, regime, shift, shift_names, bundle)
    policy = microscopic_policy(thresholds)
    output = {}
    for index, ((seed, condition), row) in enumerate(zip(tasks, rows)):
        decision = policy.decide(
            V5DecisionInputs(
                row["case_id"], row["regime"], row["shift_class"],
                row["domain_shift_score"], float(prediction.success_probability[index]),
                float(prediction.analytical_benefit[index]),
                float(prediction.corrected_microscopic_benefit[index]),
                float(prediction.microscopic_success_probability[index]),
                float(prediction.microscopic_safety_probability_upper[index]),
                bool(row["adaptive_counterfactual"]["legal"]),
            )
        )
        output[f"v6-final-{condition['id']}-s{seed}"] = bool(decision.intervene)
    return output


def _metric(rows: list[dict], mask: list[bool]) -> dict:
    return selective_metrics(rows, mask)


def run(workers: int = 4) -> Path:
    existing = OUTPUT / "summary.json"
    if existing.is_file():
        verify_frozen()
        print(existing)
        return existing
    before = verify_frozen()
    policy = load_policy()
    config = yaml.safe_load((ROOT / "configs/v6/micro_design.yaml").read_text())
    preregistration = yaml.safe_load((ROOT / "configs/v6/preregistration.yaml").read_text())
    development_seeds = set(config["development_seeds"])
    final_seeds = set(config["final_holdout_seeds"])
    if development_seeds & final_seeds:
        raise RuntimeError("v6 microscopic holdout seed leaked into development")
    tasks = [
        (int(seed), dict(condition))
        for seed in config["final_holdout_seeds"]
        for condition in config["condition_templates"]
    ]
    if len(tasks) != int(config["final_holdout_pair_count"]):
        raise RuntimeError("v6 microscopic holdout pair count mismatch")
    development_manifest = json.loads(
        (ROOT / "artifacts/studies/v6_micro_model_selection/training_manifest.json").read_text()
    )
    development_ids = {
        case_id
        for values in development_manifest["case_ids"].values()
        for case_id in values
    }
    final_ids = {f"v6-final-{condition['id']}-s{seed}" for seed, condition in tasks}
    if final_ids & development_ids:
        raise RuntimeError("v6 microscopic final case ID leaked into training")
    analytical = _analytical_predictions(config, tasks)
    completed = {
        row["case_id"]: row
        for row in (json.loads(CHECKPOINT.read_text()) if CHECKPOINT.is_file() else [])
    }
    with tempfile.TemporaryDirectory(prefix="concordia-v6-final-") as temporary:
        directory = Path(temporary)
        networks = {}
        for condition in config["condition_templates"]:
            key = (condition["topology"], condition["perturbation"])
            if key not in networks:
                networks[key] = build_v6_network(directory, *key)
        pending = []
        for seed, condition in tasks:
            case_id = f"v6-final-{condition['id']}-s{seed}"
            if case_id in completed:
                continue
            network, metadata = networks[(condition["topology"], condition["perturbation"])]
            pending.append(
                {
                    "network": str(network),
                    "metadata": metadata,
                    "config": config,
                    "condition": condition,
                    "seed": seed,
                    "analytical": analytical[f"{condition['id']}::{seed}"],
                    "preregistration": preregistration,
                }
            )
        if pending:
            with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
                futures = {executor.submit(_execute, task): task for task in pending}
                for future in as_completed(futures):
                    row = future.result()
                    if not all(row["pairing"].values()):
                        raise RuntimeError(f"v6 final pairing failed: {row['case_id']}")
                    completed[row["case_id"]] = row
                    if len(completed) % 10 == 0:
                        write_json(CHECKPOINT, sorted(completed.values(), key=lambda value: value["case_id"]))
    rows = sorted(completed.values(), key=lambda value: value["case_id"])
    if len(rows) != len(tasks):
        raise RuntimeError("v6 final microscopic dataset is incomplete")
    decisions = policy.decide(rows)
    selected = [decision["intervene"] for decision in decisions]
    composite, _benefit, unsafe = policy.probabilities(rows)
    stage1 = [
        row["features_pre_decision"]["analytical_success_probability"] >= policy.stage1_threshold
        for row in rows
    ]
    micro_only = [value >= policy.success_threshold for value in composite]
    micro_safety = [
        value >= policy.success_threshold and unsafe[index] <= policy.safety_threshold
        for index, value in enumerate(composite)
    ]
    v5 = _v5_decisions(tasks)
    masks = {
        "B1": [False] * len(rows),
        "B6": [True] * len(rows),
        "V5-F": [v5[row["case_id"]] for row in rows],
        "V6-A": stage1,
        "V6-M": micro_only,
        "V6-MS": micro_safety,
        "V6-C": selected if policy.conformal else micro_safety,
        "V6-F": selected,
    }
    metrics = {name: _metric(rows, mask) for name, mask in masks.items()}
    primary = metrics["V6-F"]
    latency = []
    for row in rows:
        started = time.perf_counter()
        policy.decide([row])
        latency.append(time.perf_counter() - started)
    eligible = claim_allowed(
        primary,
        minimum_interventions=int(preregistration["targets"]["micro_interventions"]),
        required_precision=float(preregistration["targets"]["micro_precision"]),
    )
    if primary["precision"] < 0.60 or primary["safety_violation_count"] > 0:
        outcome = "F"
    elif primary["precision"] < 0.80:
        outcome = "P"
    elif (
        primary["coverage"] >= 0.10
        and primary["opportunity_recovery_rate"] >= 0.40
        and primary["intervention_count"] >= 30
    ):
        outcome = "S"
    else:
        outcome = "P"
    summary = {
        "complete": True,
        "study": "v6 frozen actual-SUMO microscopic final holdout",
        "untouched_before_freeze": True,
        "actual_sumo": True,
        "pair_count": len(rows),
        "actual_sumo_run_count": 2 * len(rows),
        "label_counts": dict(Counter(row["label"]["diagnostic_class"] for row in rows)),
        "policy_metrics": metrics,
        "primary_metrics": primary,
        "claim_eligible": eligible,
        "provisional_outcome": outcome,
        "seed_disjoint_from_development": not bool(final_seeds & development_seeds),
        "case_ids_absent_from_model_selection": not bool(final_ids & development_ids),
        "pairing_failure_count": sum(not all(row["pairing"].values()) for row in rows),
        "predictor_inference_p95_seconds": float(np.quantile(latency, 0.95)),
        "predictor_p95_target_met": float(np.quantile(latency, 0.95)) < 0.1,
        "freeze_manifest_hash_before": before["manifest_self_hash"],
        "freeze_manifest_hash_after": verify_frozen()["manifest_self_hash"],
        "frozen_immutable": True,
        "rl_used": False,
    }
    write_json(OUTPUT / "raw_metrics.json", rows)
    write_json(OUTPUT / "decision_log.json", decisions)
    write_json(OUTPUT / "policy_masks.json", masks)
    write_json(OUTPUT / "summary.json", summary)
    CHECKPOINT.unlink(missing_ok=True)
    print(OUTPUT / "summary.json")
    return OUTPUT / "summary.json"


if __name__ == "__main__":
    run()
