#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Mapping

import yaml

from concordia.feasibility import build_alignment_case
from concordia.micro_v6 import (
    MICRO_V6_FEATURE_GROUPS,
    MICRO_V6_FEATURE_SCHEMA,
    build_safe_micro_label,
    validate_predecision_features,
)
from v5_frozen import load_deployment, prepare_cases
from v6_micro_sim import build_v6_network, run_v6_pair


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/studies/v6_micro_dataset"
CHECKPOINT = OUTPUT / "development_checkpoint.json"


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _role(seed: int, config: Mapping[str, object]) -> str:
    for role, seeds in config["development_roles"].items():
        if seed in seeds:
            return str(role)
    raise RuntimeError(f"unregistered v6 development seed: {seed}")


def _compact(run: dict) -> dict:
    return {
        key: value
        for key, value in run.items()
        if key not in {"features_pre_decision", "predecision_series", "condition"}
    }


def _execute_task(task: dict) -> dict:
    network = Path(task["network"])
    baseline, adaptive = run_v6_pair(
        network,
        task["metadata"],
        task["config"],
        task["condition"],
        int(task["seed"]),
        task["analytical"],
    )
    prereg = task["preregistration"]["label"]
    label = build_safe_micro_label(
        baseline,
        adaptive,
        minimum_relative_ttt_gain=float(prereg["minimum_relative_ttt_gain"]),
        safety_margin=float(prereg["safety_cvar_drac_margin"]),
        regret_limit=float(prereg["regret_limit"]),
    )
    condition = task["condition"]
    case_id = f"v6-dev-{condition['id']}-s{task['seed']}"
    record = {
        "case_id": case_id,
        "seed": int(task["seed"]),
        "template_id": condition["id"],
        "development_role": task["role"],
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
            "same_route_file_hash": baseline["route_file_hash"]
            == adaptive["route_file_hash"],
            "common_random_numbers": True,
        },
    }
    validate_predecision_features(record)
    if not all(record["pairing"].values()):
        raise RuntimeError(f"v6 pairing contract failed: {case_id}")
    return record


def _analytical_predictions(config: dict, tasks: list[tuple[int, dict]]) -> dict:
    cases = []
    for seed, condition in tasks:
        scenario = (
            "two_route" if condition["topology"] == "real_like" else condition["topology"]
        )
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
                source_split="v6_micro_development_predecision_screening",
            )
        )
    regime, shift, shift_names, bundle, _thresholds = load_deployment()
    _rows, prediction = prepare_cases(cases, regime, shift, shift_names, bundle)
    return {
        f"{condition['id']}::{seed}": {
            "success_probability": float(prediction.success_probability[index]),
            "predicted_ttt_gain": float(prediction.analytical_benefit[index]),
            "role": "candidate_screening_feature_only",
        }
        for index, (seed, condition) in enumerate(tasks)
    }


def run(*, pilot: bool = False, workers: int = 4) -> Path:
    existing = OUTPUT / "raw_metrics.json"
    if (ROOT / "configs/v6/frozen_micro_model.yaml").is_file():
        if not existing.is_file():
            raise RuntimeError("v6 is frozen but its micro dataset is missing")
        print(existing)
        return existing
    config = yaml.safe_load((ROOT / "configs/v6/micro_design.yaml").read_text())
    prereg = yaml.safe_load((ROOT / "configs/v6/preregistration.yaml").read_text())
    final = set(config["final_holdout_seeds"])
    development = set(config["development_seeds"])
    if final & development:
        raise RuntimeError("v6 final micro seed leaked into development")
    all_pairs = [
        (int(seed), dict(condition))
        for seed in config["development_seeds"]
        for condition in config["condition_templates"]
    ]
    target_pairs = all_pairs[: len(config["condition_templates"])] if pilot else all_pairs
    completed = {
        row["case_id"]: row
        for row in (json.loads(CHECKPOINT.read_text()) if CHECKPOINT.is_file() else [])
    }
    analytical = _analytical_predictions(config, all_pairs)
    with tempfile.TemporaryDirectory(prefix="concordia-v6-networks-") as temporary:
        directory = Path(temporary)
        networks = {}
        for condition in config["condition_templates"]:
            key = (condition["topology"], condition["perturbation"])
            if key not in networks:
                network, metadata = build_v6_network(directory, *key)
                networks[key] = (network, metadata)
        pending = []
        for seed, condition in target_pairs:
            case_id = f"v6-dev-{condition['id']}-s{seed}"
            if case_id in completed:
                continue
            network, metadata = networks[
                (condition["topology"], condition["perturbation"])
            ]
            pending.append(
                {
                    "network": str(network),
                    "metadata": metadata,
                    "config": config,
                    "condition": condition,
                    "seed": seed,
                    "role": _role(seed, config),
                    "analytical": analytical[f"{condition['id']}::{seed}"],
                    "preregistration": prereg,
                }
            )
        if pending:
            with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
                future_map = {
                    executor.submit(_execute_task, task): task for task in pending
                }
                for future in as_completed(future_map):
                    row = future.result()
                    completed[row["case_id"]] = row
                    if len(completed) % 10 == 0:
                        _write(CHECKPOINT, sorted(completed.values(), key=lambda value: value["case_id"]))
        _write(CHECKPOINT, sorted(completed.values(), key=lambda value: value["case_id"]))
    if pilot:
        pilot_seed = int(config["development_seeds"][0])
        pilot_rows = [
            completed[f"v6-dev-{condition['id']}-s{pilot_seed}"]
            for condition in config["condition_templates"]
        ]
        pilot_path = OUTPUT / "pilot_summary.json"
        _write(
            pilot_path,
            {
                "complete": True,
                "pair_count": len(pilot_rows),
                "safe_micro_success_count": sum(
                    row["label"]["safe_micro_success"] for row in pilot_rows
                ),
                "diagnostic_counts": dict(
                    Counter(row["label"]["diagnostic_class"] for row in pilot_rows)
                ),
                "design_changed_after_pilot": False,
            },
        )
        print(pilot_path)
        return pilot_path
    rows = sorted(completed.values(), key=lambda value: value["case_id"])
    if len(rows) != int(config["development_pair_count"]):
        raise RuntimeError(
            f"v6 micro development count mismatch: {len(rows)} != {config['development_pair_count']}"
        )
    if any(int(row["seed"]) in final for row in rows):
        raise RuntimeError("v6 final micro seed entered development artifact")
    roles = {
        role: sorted(row["case_id"] for row in rows if row["development_role"] == role)
        for role in config["development_roles"]
    }
    summary = {
        "complete": True,
        "study": "v6 actual-SUMO seed-disjoint SafeMicroSuccess development dataset",
        "pair_count": len(rows),
        "actual_sumo_run_count": 2 * len(rows),
        "role_counts": dict(Counter(row["development_role"] for row in rows)),
        "safe_micro_success_count": sum(row["label"]["safe_micro_success"] for row in rows),
        "safe_micro_success_rate": sum(row["label"]["safe_micro_success"] for row in rows)
        / len(rows),
        "diagnostic_counts": dict(Counter(row["label"]["diagnostic_class"] for row in rows)),
        "future_state_leakage_count": 0,
        "pairing_failure_count": 0,
        "final_holdout_materialized": False,
        "final_holdout_seeds": config["final_holdout_seeds"],
        "feature_schema_hash": hashlib.sha256(
            json.dumps(MICRO_V6_FEATURE_SCHEMA).encode()
        ).hexdigest(),
        "raw_metrics_hash": _sha(existing) if existing.is_file() else None,
        "rl_used": False,
    }
    _write(existing, rows)
    summary["raw_metrics_hash"] = _sha(existing)
    _write(OUTPUT / "dataset_summary.json", summary)
    _write(OUTPUT / "split_manifest.json", {"roles": roles, "final_seeds": config["final_holdout_seeds"]})
    _write(
        OUTPUT / "feature_schema.json",
        {
            "features": MICRO_V6_FEATURE_SCHEMA,
            "groups": MICRO_V6_FEATURE_GROUPS,
            "hash": summary["feature_schema_hash"],
        },
    )
    CHECKPOINT.unlink(missing_ok=True)
    print(existing)
    return existing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args()
    run(pilot=arguments.pilot, workers=arguments.workers)


if __name__ == "__main__":
    main()
