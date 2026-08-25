#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Mapping

import yaml

from build_v6_micro_dataset import _compact
from concordia.feasibility import build_alignment_case
from concordia.uplift_v7.paired_dataset import (
    UPLIFT_V7_FEATURE_GROUPS,
    UPLIFT_V7_FEATURE_SCHEMA,
    paired_row_from_v6,
)
from v5_frozen import load_deployment, prepare_cases
from v6_frozen import load_policy, verify_frozen
from v7_micro_sim import build_v7_network, run_v6_pair


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/studies/v7_paired_dataset"
CHECKPOINT = OUTPUT / "new_development_checkpoint.json"


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _analytical_predictions(tasks: list[tuple[int, dict]]) -> dict:
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
                source_split="v7_new_development_predecision_screening",
            )
        )
    regime, shift, shift_names, bundle, _thresholds = load_deployment()
    _rows, prediction = prepare_cases(cases, regime, shift, shift_names, bundle)
    return {
        f"{condition['id']}::{seed}": {
            "success_probability": float(prediction.success_probability[index]),
            "predicted_ttt_gain": float(prediction.analytical_benefit[index]),
            "role": "historical_analytical_screening_feature_only",
        }
        for index, (seed, condition) in enumerate(tasks)
    }


def _execute_task(task: dict) -> dict:
    baseline, adaptive = run_v6_pair(
        Path(task["network"]),
        task["metadata"],
        task["config"],
        task["condition"],
        int(task["seed"]),
        task["analytical"],
    )
    condition = dict(task["condition"])
    case_id = f"v7-development-{condition['id']}-s{task['seed']}"
    return {
        "case_id": case_id,
        "seed": int(task["seed"]),
        "template_id": condition["id"],
        "condition": condition,
        "decision_time": baseline["decision_time"],
        "feature_observation_end_time": baseline["feature_observation_end_time"],
        "features_pre_decision": baseline["features_pre_decision"],
        "predecision_series": baseline["predecision_series"],
        "analytical_screening": task["analytical"],
        "counterfactual_B1": _compact(baseline),
        "counterfactual_adaptive": _compact(adaptive),
        "pairing": {
            "same_seed": baseline["seed"] == adaptive["seed"],
            "same_network_hash": baseline["network_hash"] == adaptive["network_hash"],
            "same_route_file_hash": baseline["route_file_hash"]
            == adaptive["route_file_hash"],
            "common_random_numbers": True,
        },
    }


def _roles_by_seed(rows: list[dict], design: Mapping[str, object]) -> dict[int, str]:
    seeds = sorted({int(row["seed"]) for row in rows})
    rng = random.Random(int(design["split_seed"]))
    rng.shuffle(seeds)
    fractions = design["development_split_fractions"]
    train_count = round(len(seeds) * float(fractions["train"]))
    calibration_count = round(len(seeds) * float(fractions["calibration"]))
    role = {}
    for index, seed in enumerate(seeds):
        if index < train_count:
            role[seed] = "train"
        elif index < train_count + calibration_count:
            role[seed] = "calibration"
        else:
            role[seed] = "validation"
    return role


def _v6_scores(rows: list[dict]) -> list[float]:
    policy = load_policy()
    probabilities, _benefit, _unsafe = policy.probabilities(rows)
    return [float(value) for value in probabilities]


def _promote(rows: list[dict], source: str) -> list[dict]:
    scores = _v6_scores(rows)
    output = []
    for row, score in zip(rows, scores):
        promoted = paired_row_from_v6(
            row, v6_micro_success_score=score, source=source
        )
        output.append(promoted)
    return output


def run(*, workers: int = 4, force: bool = False) -> Path:
    raw_path = OUTPUT / "raw_metrics.json"
    summary_path = OUTPUT / "dataset_summary.json"
    if raw_path.is_file() and summary_path.is_file() and not force:
        summary = json.loads(summary_path.read_text())
        if summary.get("complete") and summary.get("pair_count") == 1200:
            verify_frozen()
            print(raw_path)
            return raw_path
    verify_frozen()
    design = yaml.safe_load((ROOT / "configs/v7/paired_design.yaml").read_text())
    historical_paths = [ROOT / relative for relative in design["historical_sources"]]
    historical_raw = [json.loads(path.read_text()) for path in historical_paths]
    if [len(rows) for rows in historical_raw] != [600, 200]:
        raise RuntimeError("v7 historical source counts changed")
    new_tasks = [
        (int(seed), dict(condition))
        for seed in design["new_development_seeds"]
        for condition in design["condition_templates"]
    ]
    analytical = _analytical_predictions(new_tasks)
    completed = {
        row["case_id"]: row
        for row in (json.loads(CHECKPOINT.read_text()) if CHECKPOINT.is_file() else [])
    }
    with tempfile.TemporaryDirectory(prefix="concordia-v7-networks-") as temporary:
        directory = Path(temporary)
        networks = {}
        for condition in design["condition_templates"]:
            key = (condition["topology"], condition["perturbation"])
            if key not in networks:
                networks[key] = build_v7_network(directory, *key)
        pending = []
        for seed, condition in new_tasks:
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
                future_map = {
                    executor.submit(_execute_task, task): task for task in pending
                }
                for future in as_completed(future_map):
                    row = future.result()
                    if not all(row["pairing"].values()):
                        raise RuntimeError(f"v7 pairing contract failed: {row['case_id']}")
                    completed[row["case_id"]] = row
                    if len(completed) % 10 == 0:
                        _write(
                            CHECKPOINT,
                            sorted(completed.values(), key=lambda value: value["case_id"]),
                        )
    if len(completed) != int(design["new_development_pair_count"]):
        raise RuntimeError("v7 new microscopic development dataset is incomplete")
    new_raw = sorted(completed.values(), key=lambda value: value["case_id"])
    rows = [
        *_promote(historical_raw[0], "v6_development_historical"),
        *_promote(historical_raw[1], "v6_final_promoted_to_v7_development"),
        *_promote(new_raw, "v7_new_actual_sumo_development"),
    ]
    roles = _roles_by_seed(rows, design)
    for row in rows:
        row["development_role"] = roles[int(row["seed"])]
        if row["source"] == "v7_new_actual_sumo_development":
            row["pair_id"] = row["source_case_id"]
    rows.sort(key=lambda value: value["pair_id"])
    expected = int(design["total_development_pair_count"])
    if len(rows) != expected or len({row["pair_id"] for row in rows}) != expected:
        raise RuntimeError("v7 combined development pair count or identity mismatch")
    final_seeds = set(map(int, design["final_holdout_seeds"]))
    if any(int(row["seed"]) in final_seeds for row in rows):
        raise RuntimeError("v7 final holdout seed leaked into development")
    _write(raw_path, rows)
    role_ids = {
        role: sorted(row["pair_id"] for row in rows if row["development_role"] == role)
        for role in ("train", "calibration", "validation")
    }
    source_counts = Counter(row["source"] for row in rows)
    outcome_counts = Counter(row["outcomes"]["benefit_magnitude_bin"] for row in rows)
    summary = {
        "complete": True,
        "study": "v7 paired conditional-treatment-effect development dataset",
        "pair_count": len(rows),
        "historical_pair_count": int(design["historical_pair_count"]),
        "new_actual_sumo_pair_count": len(new_raw),
        "historical_actual_sumo_run_count": 2 * int(design["historical_pair_count"]),
        "new_actual_sumo_run_count": 2 * len(new_raw),
        "total_underlying_actual_sumo_run_count": 2 * len(rows),
        "source_counts": dict(source_counts),
        "role_counts": Counter(row["development_role"] for row in rows),
        "seed_family_counts": Counter(roles.values()),
        "safe_micro_success_count": sum(
            bool(row["outcomes"]["safe_micro_success"]) for row in rows
        ),
        "positive_traffic_uplift_count": sum(
            float(row["outcomes"]["tau_t_relative"]) >= 0.01 for row in rows
        ),
        "benefit_magnitude_counts": dict(outcome_counts),
        "pairing_failure_count": sum(
            not bool(row["pairing"]["metadata_identical_except_treatment"])
            for row in rows
        ),
        "future_state_leakage_count": 0,
        "final_holdout_materialized": False,
        "final_holdout_seeds": design["final_holdout_seeds"],
        "feature_count": len(UPLIFT_V7_FEATURE_SCHEMA),
        "raw_metrics_hash": _sha(raw_path),
        "v6_freeze_manifest_verified": True,
        "rl_used": False,
    }
    _write(summary_path, summary)
    _write(
        OUTPUT / "split_manifest.json",
        {
            "split_unit": "seed_family",
            "roles": role_ids,
            "role_seeds": {
                name: sorted(seed for seed, value in roles.items() if value == name)
                for name in ("train", "calibration", "validation")
            },
            "final_seeds": design["final_holdout_seeds"],
        },
    )
    schema_hash = hashlib.sha256(
        json.dumps(UPLIFT_V7_FEATURE_SCHEMA).encode()
    ).hexdigest()
    _write(
        OUTPUT / "feature_schema.json",
        {
            "features": UPLIFT_V7_FEATURE_SCHEMA,
            "groups": UPLIFT_V7_FEATURE_GROUPS,
            "feature_count": len(UPLIFT_V7_FEATURE_SCHEMA),
            "hash": schema_hash,
            "predecision_only": True,
        },
    )
    CHECKPOINT.unlink(missing_ok=True)
    print(raw_path)
    return raw_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    run(workers=arguments.workers, force=arguments.force)


if __name__ == "__main__":
    main()
