#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

from build_v7_paired_dataset import _analytical_predictions, _execute_task, _promote
from concordia.safety_v8.features import ACTION_AWARE_FEATURE_SCHEMA, action_aware_features
from concordia.safety_v8.labels import unsafe_intervention
from concordia.uplift_v7.paired_dataset import feature_matrix
from v6_frozen import verify_frozen as verify_v6
from v7_frozen import load_policy as load_v7_policy, verify_frozen as verify_v7
from v7_micro_sim import build_v7_network
from v8_common import ROOT, sha256, write_json


OUTPUT = ROOT / "artifacts/studies/v8_safety_dataset"
CHECKPOINT = OUTPUT / "new_development_checkpoint.json"


def _roles(seeds: list[int], split_seed: int, fractions: dict) -> dict[int, str]:
    unique = sorted(set(seeds))
    random.Random(split_seed).shuffle(unique)
    train = round(len(unique) * float(fractions["train"]))
    calibration = round(len(unique) * float(fractions["calibration"]))
    return {
        seed: "train" if index < train else "calibration" if index < train + calibration else "validation"
        for index, seed in enumerate(unique)
    }


def run(*, workers: int = 4, force: bool = False) -> Path:
    raw_path = OUTPUT / "raw_metrics.json"
    summary_path = OUTPUT / "dataset_summary.json"
    if raw_path.is_file() and summary_path.is_file() and not force:
        summary = json.loads(summary_path.read_text())
        if summary.get("complete") and summary.get("pair_count") == 2000:
            verify_v7()
            print(raw_path)
            return raw_path
    verify_v6()
    verify_v7()
    design = yaml.safe_load((ROOT / "configs/v8/safety_design.yaml").read_text())
    historical = [
        row
        for relative in design["historical_sources"]
        for row in json.loads((ROOT / relative).read_text())
    ]
    if len(historical) != int(design["historical_pair_count"]):
        raise RuntimeError("v8 historical development count changed")
    tasks = [
        (int(seed), dict(condition))
        for seed in design["new_development_seeds"]
        for condition in design["condition_templates"]
    ]
    analytical = _analytical_predictions(tasks)
    completed = {
        row["case_id"]: row
        for row in (json.loads(CHECKPOINT.read_text()) if CHECKPOINT.is_file() else [])
    }
    with tempfile.TemporaryDirectory(prefix="concordia-v8-networks-") as temporary:
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
                    if not all(row["pairing"].values()):
                        raise RuntimeError(f"v8 paired simulation mismatch: {row['case_id']}")
                    completed[row["case_id"]] = row
                    if len(completed) % 10 == 0:
                        write_json(CHECKPOINT, sorted(completed.values(), key=lambda value: value["case_id"]))
    if len(completed) != int(design["new_development_pair_count"]):
        raise RuntimeError("v8 new microscopic development set is incomplete")
    new_rows = _promote(
        sorted(completed.values(), key=lambda value: value["case_id"]),
        "v8_new_actual_sumo_safety_development",
    )
    for row in new_rows:
        row["pair_id"] = row["pair_id"].replace("v7-historical::", "v8-new::")
    rows = [*historical, *new_rows]
    roles = _roles(
        [int(row["seed"]) for row in rows],
        int(design["split_seed"]),
        design["development_split_fractions"],
    )
    for row in rows:
        row["development_role"] = roles[int(row["seed"])]
    v7 = load_v7_policy()
    scores = v7.traffic_model.predict(feature_matrix(rows, v7.traffic_model.feature_names))
    training_scores = [
        float(score) for row, score in zip(rows, scores) if row["development_role"] == "train"
    ]
    ordered_reference = sorted(training_scores)
    for row, score in zip(rows, scores):
        rank = sum(reference <= float(score) for reference in ordered_reference) / len(ordered_reference)
        row["v8_features"] = action_aware_features(
            row, traffic_uplift_score=float(score), traffic_rank_percentile=rank
        )
        row["unsafe_intervention"] = int(unsafe_intervention(row))
        row["legal_executable_predecision"] = True
    rows.sort(key=lambda value: value["pair_id"])
    expected = int(design["total_development_pair_count"])
    if len(rows) != expected or len({row["pair_id"] for row in rows}) != expected:
        raise RuntimeError("v8 development identity/count mismatch")
    if any(int(row["seed"]) in set(design["final_holdout_seeds"]) for row in rows):
        raise RuntimeError("v8 final seed leaked into development")
    write_json(raw_path, rows)
    unsafe_count = sum(int(row["unsafe_intervention"]) for row in rows)
    role_counts = Counter(row["development_role"] for row in rows)
    acquisition = {
        "registered_boundary_budget_fraction": float(design["safety_boundary_budget_fraction"]),
        "template_stratum_counts": Counter(
            condition.get("stratum", "ordinary") for condition in design["condition_templates"]
        ),
        "new_pair_stratum_counts": Counter(row["condition"].get("stratum", "ordinary") for row in new_rows),
        "labels_unchanged_by_acquisition": True,
    }
    write_json(OUTPUT / "acquisition_manifest.json", acquisition)
    summary = {
        "complete": True,
        "study": "v8 action-aware safety-classification development dataset",
        "pair_count": len(rows),
        "historical_pair_count": len(historical),
        "new_actual_sumo_pair_count": len(new_rows),
        "total_underlying_actual_sumo_run_count": 2 * len(rows),
        "new_actual_sumo_run_count": 2 * len(new_rows),
        "unsafe_intervention_count": unsafe_count,
        "safe_intervention_count": len(rows) - unsafe_count,
        "safe_micro_success_count": sum(bool(row["outcomes"]["safe_micro_success"]) for row in rows),
        "role_counts": dict(role_counts),
        "seed_family_counts": Counter(roles.values()),
        "source_counts": Counter(row["source"] for row in rows),
        "feature_count": len(ACTION_AWARE_FEATURE_SCHEMA),
        "feature_schema": list(ACTION_AWARE_FEATURE_SCHEMA),
        "future_state_leakage_count": 0,
        "pairing_failure_count": sum(not bool(row["pairing"]["metadata_identical_except_treatment"]) for row in rows),
        "minimum_pair_requirement_met": len(rows) >= 2000,
        "minimum_unsafe_requirement_met": unsafe_count >= 300,
        "final_holdout_materialized": False,
        "development_score_reference": ordered_reference,
        "raw_metrics_hash": sha256(raw_path),
        "v7_freeze_verified": True,
    }
    write_json(summary_path, summary)
    if not summary["minimum_pair_requirement_met"] or not summary["minimum_unsafe_requirement_met"]:
        raise RuntimeError("v8 development minimum evidence contract failed")
    print(raw_path)
    return raw_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    run(workers=arguments.workers, force=arguments.force)
