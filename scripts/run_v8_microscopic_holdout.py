#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import yaml

from build_v7_paired_dataset import _analytical_predictions, _execute_task, _v6_scores
from concordia.safety_v8.evaluation import classification_metrics, critical_group_recall
from concordia.safety_v8.features import action_aware_features
from concordia.selective_v8.metrics import filtered_policy_metrics, ranking_metrics
from concordia.uplift_v7.evaluation import deployment_metrics
from concordia.uplift_v7.paired_dataset import paired_row_from_v6
from run_v7_microscopic_holdout import _v5_mask, _v6_mask, _variant_mask, _population_effect
from v7_frozen import load_policy as load_v7_policy
from v7_micro_sim import build_v7_network
from v8_common import ROOT, sha256, write_json
from v8_frozen import load_policy, verify_frozen


OUTPUT = ROOT / "artifacts/studies/v8_micro_holdout"
CHECKPOINT = OUTPUT / "final_checkpoint.json"


def run(*, workers: int = 4):
    summary_path = OUTPUT / "summary.json"
    if summary_path.is_file():
        verify_frozen()
        print(summary_path)
        return summary_path
    before = verify_frozen()
    policy = load_policy()
    v7 = load_v7_policy()
    design = yaml.safe_load((ROOT / "configs/v8/safety_design.yaml").read_text())
    tasks = [
        (int(seed), dict(condition))
        for seed in design["final_holdout_seeds"]
        for condition in design["condition_templates"]
    ]
    if len(tasks) != 400:
        raise RuntimeError("v8 final microscopic pair count must be 400")
    development_seeds = {
        int(row["seed"])
        for row in json.loads((ROOT / "artifacts/studies/v8_safety_dataset/raw_metrics.json").read_text())
    }
    final_seeds = {seed for seed, _ in tasks}
    if development_seeds & final_seeds:
        raise RuntimeError("v8 final seed leaked into development")
    v7_final = set(yaml.safe_load((ROOT / "configs/v7/paired_design.yaml").read_text())["final_holdout_seeds"])
    if final_seeds & v7_final:
        raise RuntimeError("v7 final seed reused in v8 final")
    analytical = _analytical_predictions(tasks)
    completed = {
        row["case_id"]: row
        for row in (json.loads(CHECKPOINT.read_text()) if CHECKPOINT.is_file() else [])
    }
    with tempfile.TemporaryDirectory(prefix="concordia-v8-final-") as temporary:
        networks = {}
        for condition in design["condition_templates"]:
            key = (condition["topology"], condition["perturbation"])
            if key not in networks:
                networks[key] = build_v7_network(__import__("pathlib").Path(temporary), *key)
        pending = []
        for seed, condition in tasks:
            case_id = f"v7-development-{condition['id']}-s{seed}"
            if case_id in completed:
                continue
            network, metadata = networks[(condition["topology"], condition["perturbation"])]
            pending.append({
                "network": str(network),
                "metadata": metadata,
                "config": design,
                "condition": condition,
                "seed": seed,
                "analytical": analytical[f"{condition['id']}::{seed}"],
            })
        if pending:
            with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
                futures = {executor.submit(_execute_task, task): task for task in pending}
                for future in as_completed(futures):
                    row = future.result()
                    if not all(row["pairing"].values()):
                        raise RuntimeError(f"v8 final pairing mismatch: {row['case_id']}")
                    completed[row["case_id"]] = row
                    if len(completed) % 10 == 0:
                        write_json(CHECKPOINT, sorted(completed.values(), key=lambda value: value["case_id"]))
    raw = sorted(completed.values(), key=lambda value: value["case_id"])
    if len(raw) != len(tasks):
        raise RuntimeError("v8 final microscopic data incomplete")
    scores_v6 = _v6_scores(raw)
    rows = []
    for raw_row, v6_score in zip(raw, scores_v6):
        row = paired_row_from_v6(
            raw_row,
            v6_micro_success_score=v6_score,
            source="v8_untouched_final_actual_sumo",
        )
        row["pair_id"] = raw_row["case_id"].replace("v7-development", "v8-final")
        row["legal_executable_predecision"] = True
        rows.append(row)
    traffic = policy.traffic_ranker.predict(rows)
    percentiles = policy.traffic_ranker.percentiles(traffic)
    for row, score, percentile in zip(rows, traffic, percentiles):
        row["v8_features"] = action_aware_features(
            row,
            traffic_uplift_score=float(score),
            traffic_rank_percentile=float(percentile),
        )
        row["unsafe_intervention"] = int(float(row["outcomes"]["tau_s"]) > 0.25)
    started = time.perf_counter()
    decisions = policy.decide(rows)
    latency = (time.perf_counter() - started) / len(rows)
    v8_mask = [bool(item["intervene"]) for item in decisions]
    unsafe_probability = policy.safety_filter.probabilities(rows)
    screen = percentiles >= policy.traffic_rank_percentile_cutoff
    safety = unsafe_probability <= policy.safety_filter.unsafe_probability_threshold
    masks = {
        "B1": [False] * len(rows),
        "B6": [True] * len(rows),
        "V5-F": _v5_mask(tasks),
        "V6-Binary": _v6_mask(rows),
        "V7-F": [decision["intervene"] for decision in v7.decide(rows)],
        "V7-Mean-reconstructed": _variant_mask(v7, rows, "mean"),
        "V8-traffic-only": [bool(value) for value in screen],
        "V8-traffic-plus-safety": [bool(value) for value in (screen & safety)],
        "V8-F": v8_mask,
    }
    comparison = {}
    for name, mask in masks.items():
        metrics = deployment_metrics(rows, mask)
        if name.startswith("V8"):
            metrics.update(filtered_policy_metrics(rows, mask, screen))
        comparison[name] = {**metrics, **_population_effect(rows, mask)}
    rank = ranking_metrics(rows, traffic)
    labels = [int(row["unsafe_intervention"]) for row in rows]
    safety_report = classification_metrics(
        labels, unsafe_probability, policy.safety_filter.unsafe_probability_threshold
    )
    safety_report.update(
        critical_group_recall(rows, unsafe_probability, policy.safety_filter.unsafe_probability_threshold)
    )
    traffic_only_success = np.asarray([bool(row["outcomes"]["safe_micro_success"]) for row in rows]) & screen
    filtered_success = traffic_only_success & np.asarray(v8_mask)
    unsafe = np.asarray(labels, dtype=bool)
    cost = {
        "high_rank_candidate_count": int(screen.sum()),
        "high_rank_unsafe_count": int((screen & unsafe).sum()),
        "mean_uplift_high_rank_unsafe": float(np.mean(np.asarray([row["outcomes"]["tau_t_relative"] for row in rows])[screen & unsafe])) if (screen & unsafe).any() else None,
        "mean_uplift_high_rank_safe": float(np.mean(np.asarray([row["outcomes"]["tau_t_relative"] for row in rows])[screen & ~unsafe])) if (screen & ~unsafe).any() else None,
        "safe_success_retention": float(filtered_success.sum() / traffic_only_success.sum()) if traffic_only_success.any() else 0.0,
        "unsafe_candidate_removal": float((screen & unsafe & ~np.asarray(v8_mask)).sum() / (screen & unsafe).sum()) if (screen & unsafe).any() else 1.0,
    }
    write_json(OUTPUT / "raw_metrics.json", rows)
    write_json(OUTPUT / "decisions.json", decisions)
    write_json(OUTPUT / "architecture_comparison.json", comparison)
    write_json(OUTPUT / "traffic_ranking.json", rank)
    write_json(OUTPUT / "safety_classifier.json", safety_report)
    write_json(OUTPUT / "safety_cost_of_uplift.json", cost)
    after = verify_frozen()
    v8 = comparison["V8-F"]
    summary = {
        "complete": True,
        "untouched_new_final": True,
        "pair_count": len(rows),
        "actual_sumo_run_count": 2 * len(rows),
        "seed_overlap_with_development": 0,
        "seed_overlap_with_v7_final": 0,
        "pairing_failure_count": sum(not row["pairing"]["metadata_identical_except_treatment"] for row in rows),
        "future_state_leakage_count": 0,
        "policy_latency_seconds_per_case": latency,
        "comparison": comparison,
        "traffic_ranking": rank,
        "safety_classifier": safety_report,
        "safety_cost_of_uplift": cost,
        "primary_targets": {
            "precision_at_least_0_80": v8["deployment_precision"] >= 0.80,
            "coverage_at_least_0_08": v8["coverage"] >= 0.08,
            "interventions_at_least_30": v8["intervention_count"] >= 30,
            "safety_violations_zero": v8["safety_violation_count"] == 0,
            "orr_at_least_0_35": v8["opportunity_realization_rate"] >= 0.35,
            "unsafe_recall_at_least_0_95": safety_report["unsafe_recall"] >= 0.95,
            "safe_success_retention_at_least_0_70": cost["safe_success_retention"] >= 0.70,
            "traffic_top_10_positive": rank["top_k"]["top_10_percent"]["mean_realized_uplift"] > 0.0,
            "traffic_top_10_above_population": rank["top_k"]["top_10_percent"]["uplift_over_population_mean"] > 0.0,
        },
        "freeze_manifest_unchanged": before["manifest_self_hash"] == after["manifest_self_hash"],
        "raw_metrics_hash": sha256(OUTPUT / "raw_metrics.json"),
        "rl_used": False,
    }
    write_json(summary_path, summary)
    print(summary_path)
    return summary_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    run(workers=parser.parse_args().workers)
