#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml

from concordia.v10.integrity import assert_file_hashes
from concordia.v10.racing import MultiFidelityRacer
from concordia.v9.action_space import b6_reference_action, generate_action_library
from run_v10_racing_study import StateRolloutEvaluator, _write
from v10_micro_sim import build_v9_network


ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "artifacts/v10/freeze_manifest.json"
SEEDS = ROOT / "artifacts/v10/final_seed_manifest.json"
STUDY_DIR = ROOT / "artifacts/studies/v10_micro_holdout"


def _safe_success(outcome: dict) -> bool:
    return bool(
        float(outcome["traffic_gain"]) > 0.005
        and float(outcome["risk_delta"]) <= 0.25
        and float(outcome["maximum_regret"]) <= 0.08
        and bool(outcome["legal"])
    )


def _frozen_racing_config() -> dict:
    stage3 = yaml.safe_load((ROOT / "configs/v10/frozen_stage3.yaml").read_text())
    verification = stage3.pop("verification")
    return {
        "stage_1": yaml.safe_load((ROOT / "configs/v10/frozen_stage1.yaml").read_text()),
        "stage_2": yaml.safe_load((ROOT / "configs/v10/frozen_stage2.yaml").read_text()),
        "stage_3": stage3,
        "verification": verification,
        "selector": yaml.safe_load((ROOT / "configs/v10/frozen_selector.yaml").read_text()),
    }


def _summary(records: list[dict]) -> dict:
    interventions = [record for record in records if record["trace"]["intervene"]]
    successes = sum(record["selected_safe_success"] for record in interventions)
    safety = sum(
        float(record["selected_outcome"]["risk_delta"]) > 0.25
        for record in interventions
    )
    regret = sum(
        float(record["selected_outcome"]["maximum_regret"]) > 0.08
        for record in interventions
    )
    legal = sum(not record["selected_outcome"]["legal"] for record in interventions)
    coverage = len(interventions) / len(records)
    precision = successes / len(interventions) if interventions else 0.0
    population_gain = sum(
        float(record["selected_outcome"]["traffic_gain"])
        if record["trace"]["intervene"] else 0.0
        for record in records
    ) / len(records)
    b6_successes = sum(_safe_success(record["b6_outcome"]) for record in records)
    b6_safety = sum(float(record["b6_outcome"]["risk_delta"]) > 0.25 for record in records)
    b6_gain = sum(float(record["b6_outcome"]["traffic_gain"]) for record in records) / len(records)
    latencies = sorted(float(record["trace"]["decision_latency_seconds"]) for record in records)
    primary_s = bool(
        precision >= 0.80 and coverage >= 0.10 and len(interventions) >= 40
        and safety == 0
    )
    if (
        primary_s and precision >= 0.85 and coverage >= 0.15
        and population_gain > 0.0
    ):
        outcome = "S+"
    elif primary_s:
        outcome = "S"
    elif precision >= 0.70 and coverage > 0.0 and safety == 0:
        outcome = "P"
    else:
        outcome = "F"
    return {
        "state_count": len(records),
        "intervention_count": len(interventions),
        "safe_beneficial_intervention_count": successes,
        "precision": precision,
        "coverage": coverage,
        "safety_violation_count": safety,
        "regret_violation_count": regret,
        "legal_violation_count": legal,
        "population_mean_relative_ttt_gain": population_gain,
        "b6_safe_success_count": b6_successes,
        "b6_precision": b6_successes / len(records),
        "b6_safety_violation_count": b6_safety,
        "b6_population_mean_relative_ttt_gain": b6_gain,
        "mean_decision_latency_seconds": sum(latencies) / len(latencies),
        "p95_decision_latency_seconds": latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))],
        "decision_evaluation_seed_overlap": sum(
            record["trace"]["decision_evaluation_seed_overlap"] for record in records
        ),
        "rollout_count": sum(len(record["trace"]["rollout_results"]) for record in records),
        "final_outcome": outcome,
        "primary_targets": {
            "precision_at_least_0_80": precision >= 0.80,
            "coverage_at_least_0_10": coverage >= 0.10,
            "interventions_at_least_40": len(interventions) >= 40,
            "safety_violations_zero": safety == 0,
        },
        "final_holdout_materialized": True,
        "final_realized_outcome_cache_used": False,
        "rl_used": False,
    }


def run(*, workers: int) -> Path:
    raw_path = STUDY_DIR / "raw_metrics.json"
    summary_path = STUDY_DIR / "summary.json"
    checkpoint = STUDY_DIR / "final_checkpoint.json"
    if raw_path.exists() or summary_path.exists():
        raise RuntimeError("v10 final holdout has already completed and cannot be rerun")
    if not FREEZE.is_file() or not SEEDS.is_file():
        raise RuntimeError("v10 freeze and final seed manifest are required")
    freeze = json.loads(FREEZE.read_text())
    assert_file_hashes(freeze["frozen_file_hashes"], ROOT)
    assert_file_hashes(freeze["implementation_file_hashes"], ROOT)
    seed_manifest = json.loads(SEEDS.read_text())
    if seed_manifest["freeze_commit"] != seed_manifest["remote_main_commit_verified"]:
        raise RuntimeError("v10 final seed manifest lacks remote freeze verification")
    source = yaml.safe_load((ROOT / "configs/v9/development_design.yaml").read_text())
    specs = [
        (int(seed), dict(condition))
        for seed in seed_manifest["seed_families"]
        for condition in source["condition_templates"]
    ]
    expected_ids = [f"V10F::{condition['id']}::s{seed}" for seed, condition in specs]
    if expected_ids != seed_manifest["state_ids"] or len(specs) != 500:
        raise RuntimeError("v10 final state manifest contract failed")
    simulation_config = yaml.safe_load((ROOT / "configs/v9/development_design.yaml").read_text())
    racing_config = _frozen_racing_config()
    racer = MultiFidelityRacer(racing_config)
    actions = [action.to_dict() for action in generate_action_library()]
    b6 = b6_reference_action()
    all_actions = [*actions, b6]
    completed = {
        record["state_id"]: record
        for record in (json.loads(checkpoint.read_text()) if checkpoint.is_file() else [])
    }
    with tempfile.TemporaryDirectory(prefix="concordia-v10-final-networks-") as temporary:
        directory = Path(temporary)
        networks = {}
        for _seed, condition in specs:
            key = (condition["topology"], condition["perturbation"])
            if key not in networks:
                networks[key] = build_v9_network(directory, *key)
        with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
            for seed, condition in specs:
                state_id = f"V10F::{condition['id']}::s{seed}"
                if state_id in completed:
                    continue
                network, metadata = networks[(condition["topology"], condition["perturbation"])]
                evaluator = StateRolloutEvaluator(
                    state_id=state_id, state_seed=seed, network=network,
                    metadata=metadata, simulation_config=simulation_config,
                    condition=condition, actions=all_actions, executor=executor,
                    parameter_label="final_decision_nominal",
                )
                baseline = evaluator.baseline(
                    int(racing_config["stage_1"]["horizon_seconds"]),
                    plan_actions=actions,
                )
                started = time.perf_counter()
                trace = racer.race(
                    state_id, actions, baseline["candidate_plans"], evaluator,
                    evaluation_seed=seed,
                )
                trace["decision_latency_seconds"] = time.perf_counter() - started
                trace["normalized_network_hash_pair_count"] = (
                    evaluator.normalized_network_hash_pair_count
                )
                selected = trace["selected_action_id"]
                _baseline, outcomes = evaluator.actual_actions(
                    [selected, b6["action_id"]], horizon_seconds=600,
                    evaluation_seed=seed,
                )
                evaluated_actions = [b6["action_id"]]
                if selected != "A00_NULL_B1":
                    evaluated_actions.append(selected)
                incomplete = [
                    action_id for action_id in evaluated_actions
                    if (
                        outcomes[action_id]["arrived_b1"]
                        != outcomes[action_id]["generated_b1"]
                        or outcomes[action_id]["arrived_action"]
                        != outcomes[action_id]["generated_action"]
                    )
                ]
                if incomplete:
                    raise RuntimeError(
                        f"v10 final TTT horizon did not clear vehicles: {incomplete}"
                    )
                selected_outcome = outcomes[selected]
                record = {
                    "state_id": state_id,
                    "seed": seed,
                    "template_id": condition["id"],
                    "condition": condition,
                    "trace": trace,
                    "selected_outcome": selected_outcome,
                    "selected_safe_success": (
                        _safe_success(selected_outcome) if trace["intervene"] else False
                    ),
                    "b6_outcome": outcomes[b6["action_id"]],
                }
                completed[state_id] = record
                _write(checkpoint, [completed[state] for state in expected_ids if state in completed])
                print(
                    f"v10 final {len(completed)}/500 {state_id} selected={selected}",
                    flush=True,
                )
    records = [completed[state_id] for state_id in expected_ids]
    _write(raw_path, records)
    _write(summary_path, _summary(records))
    checkpoint.unlink(missing_ok=True)
    print(summary_path)
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    arguments = parser.parse_args()
    run(workers=arguments.workers)


if __name__ == "__main__":
    main()
