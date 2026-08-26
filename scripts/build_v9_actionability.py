#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Mapping

import yaml

from build_v6_micro_dataset import _compact
from build_v7_paired_dataset import _analytical_predictions
from concordia.uplift_v7.outcomes import paired_treatment_outcomes
from concordia.uplift_v7.paired_dataset import enrich_predecision_features
from concordia.v9.action_features import (
    ACTION_FEATURE_SCHEMA,
    STATE_ACTION_FEATURE_SCHEMA,
    build_action_features,
    state_action_features,
)
from concordia.v9.action_space import (
    AdaptiveAction,
    b6_reference_action,
    generate_action_library,
)
from concordia.v9.oracle import oracle_actionability, oracle_for_state
from v6_frozen import load_policy, verify_frozen
from v9_micro_sim import build_v9_network, run_v9_action


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/studies/v9_actionability"
CHECKPOINT = OUTPUT / "development_checkpoint.json"


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _state_id(seed: int, condition: Mapping[str, object]) -> str:
    return f"V9D::{condition['id']}::s{seed}"


def _is_exhaustive(condition: Mapping[str, object]) -> bool:
    return int(str(condition["id"])[1:]) % 5 == 1


def _actual_actions(state_id: str, library: tuple[AdaptiveAction, ...]) -> list[AdaptiveAction]:
    non_null = list(library[1:])
    if state_id.startswith("V9D") and int(state_id.split("::")[1][1:]) % 5 == 1:
        return non_null
    offset = int(hashlib.sha256(state_id.encode()).hexdigest()[:8], 16) % len(non_null)
    return [non_null[(offset + 3 * index) % len(non_null)] for index in range(8)]


def _execute_state(task: dict) -> dict:
    condition = dict(task["condition"])
    seed = int(task["seed"])
    state_id = _state_id(seed, condition)
    library = tuple(AdaptiveAction.from_dict(value) for value in task["library"])
    actual_actions = _actual_actions(state_id, library)
    b6 = b6_reference_action()
    plan_actions = [*library, b6]
    baseline = run_v9_action(
        Path(task["network"]), task["metadata"], task["config"], condition,
        seed, seed, library[0], task["analytical"], plan_actions=plan_actions,
    )
    action_runs = []
    for action in [*actual_actions, b6]:
        action_runs.append(
            run_v9_action(
                Path(task["network"]), task["metadata"], task["config"], condition,
                seed, seed, action, task["analytical"],
            )
        )
    for run in action_runs:
        if (
            run["network_hash"] != baseline["network_hash"]
            or run["route_file_hash"] != baseline["route_file_hash"]
            or run["feature_observation_end_time"] != baseline["feature_observation_end_time"]
        ):
            raise RuntimeError(f"common-state pairing failed for {state_id}/{run['policy']}")
    return {
        "state_id": state_id,
        "seed": seed,
        "template_id": condition["id"],
        "condition": condition,
        "exhaustive_oracle": _is_exhaustive(condition),
        "analytical": task["analytical"],
        "features_pre_decision": baseline["features_pre_decision"],
        "candidate_plans": baseline["candidate_plans"],
        "baseline": _compact(baseline),
        "action_runs": [_compact(run) for run in action_runs],
    }


def _roles(seeds: list[int], design: Mapping[str, object]) -> dict[int, str]:
    ordered = sorted(seeds)
    random.Random(int(design["development_split_seed"])).shuffle(ordered)
    fractions = design["development_split_by_seed_family"]
    train_count = round(len(ordered) * float(fractions["train"]))
    calibration_count = round(len(ordered) * float(fractions["calibration"]))
    return {
        seed: "train" if index < train_count else "calibration" if index < train_count + calibration_count else "validation"
        for index, seed in enumerate(ordered)
    }


def _action_rows(states: list[dict], design: Mapping[str, object]) -> list[dict]:
    policy = load_policy()
    state_inputs = [
        {
            "case_id": state["state_id"],
            "features_pre_decision": state["features_pre_decision"],
        }
        for state in states
    ]
    v6_scores, _benefit, _unsafe = policy.probabilities(state_inputs)
    roles = _roles([int(seed) for seed in design["development_seeds"]], design)
    library = {action.action_id: action.to_dict() for action in generate_action_library()}
    library[b6_reference_action()["action_id"]] = b6_reference_action()
    rows = []
    for state, v6_score in zip(states, v6_scores):
        baseline = state["baseline"]
        baseline_counterfactual = {
            key: value
            for key, value in baseline.items()
            if key not in {"action", "action_plan", "candidate_plans"}
        }
        state_features = enrich_predecision_features(
            state["features_pre_decision"], state["condition"],
            v6_micro_success_score=float(v6_score), number_route_alternatives=3,
        )
        null_outcomes = paired_treatment_outcomes(
            baseline, baseline, minimum_relative_uplift=0.005,
            safety_margin=0.25, regret_limit=0.08,
        ).to_dict()
        all_runs = [
            {
                **baseline,
                "policy": "A00_NULL_B1",
                "action": library["A00_NULL_B1"],
                "action_plan": state["candidate_plans"]["A00_NULL_B1"],
                "_outcomes": null_outcomes,
            },
            *[{**run, "_outcomes": paired_treatment_outcomes(
                baseline, run, minimum_relative_uplift=0.005,
                safety_margin=0.25, regret_limit=0.08,
            ).to_dict()} for run in state["action_runs"]],
        ]
        for run in all_runs:
            action_id = str(run["policy"])
            action = library[action_id]
            plan = state["candidate_plans"][action_id]
            action_features = build_action_features(action, plan)
            rows.append({
                "row_id": f"{state['state_id']}::{action_id}",
                "state_id": state["state_id"],
                "seed": int(state["seed"]),
                "template_id": state["template_id"],
                "condition": state["condition"],
                "development_role": roles[int(state["seed"])],
                "exhaustive_oracle": bool(state["exhaustive_oracle"]),
                "action_id": action_id,
                "action": action,
                "state_features": state_features,
                "action_features": action_features,
                "state_action_features": state_action_features(state_features, action_features),
                "outcomes": run["_outcomes"],
                "counterfactual_B1": baseline_counterfactual,
                "counterfactual_action": {
                    key: value
                    for key, value in run.items()
                    if key not in {"_outcomes", "action_plan", "candidate_plans"}
                },
                "pairing": {
                    "same_seed": int(baseline["seed"]) == int(run["seed"]),
                    "same_network_hash": baseline["network_hash"] == run["network_hash"],
                    "same_route_file_hash": baseline["route_file_hash"] == run["route_file_hash"],
                    "same_predecision_features": True,
                    "common_random_numbers": True,
                },
            })
    return sorted(rows, key=lambda row: row["row_id"])


def _oracle_report(rows: list[dict]) -> dict:
    experimental = [row for row in rows if row["action_id"] != "B6_ALWAYS_ON_REFERENCE"]
    all_evaluated = oracle_actionability(experimental)
    exhaustive = [row for row in experimental if row["exhaustive_oracle"]]
    exhaustive_report = oracle_actionability(exhaustive)
    b6_rows = [row for row in rows if row["action_id"] == "B6_ALWAYS_ON_REFERENCE"]
    b6_success = sum(
        bool(row["outcomes"]["safe_micro_success"]) for row in b6_rows
    ) / max(len(b6_rows), 1)
    groups = defaultdict(list)
    for row in experimental:
        groups[row["state_id"]].append(row)
    interior = 0
    oracle_states = []
    for state_id, values in sorted(groups.items()):
        report = oracle_for_state(values)
        selected = next(
            (row for row in values if row["action_id"] == report["oracle_safe_beneficial_action_id"]),
            None,
        )
        report["oracle_intensity"] = float(selected["action"]["reroute_fraction"]) if selected else 0.0
        report["interior_intensity"] = bool(selected and 0.05 < report["oracle_intensity"] < 0.30)
        interior += int(report["interior_intensity"])
        oracle_states.append(report)
    return {
        "gate_A_threshold": 0.40,
        "all_evaluated_actionability": all_evaluated["oracle_actionability_rate"],
        "exhaustive_oracle_actionability": exhaustive_report["oracle_actionability_rate"],
        "exhaustive_state_count": exhaustive_report["state_count"],
        "single_B6_safe_success_rate": b6_success,
        "multi_action_improvement_over_B6": all_evaluated["oracle_actionability_rate"] - b6_success,
        "gate_A_pass": all_evaluated["oracle_actionability_rate"] >= 0.40,
        "interior_optimum_state_count": interior,
        "states": oracle_states,
    }


def run(*, workers: int = 8, force: bool = False, limit: int | None = None) -> Path:
    verify_frozen()
    design = yaml.safe_load((ROOT / "configs/v9/development_design.yaml").read_text())
    library = generate_action_library()
    task_specs = [
        (int(seed), dict(condition))
        for seed in design["development_seeds"]
        for condition in design["condition_templates"]
    ]
    if limit is not None:
        task_specs = task_specs[:limit]
    analytical = _analytical_predictions(task_specs)
    completed = {
        row["state_id"]: row
        for row in (json.loads(CHECKPOINT.read_text()) if CHECKPOINT.is_file() and not force else [])
    }
    with tempfile.TemporaryDirectory(prefix="concordia-v9-networks-") as temporary:
        directory = Path(temporary)
        networks = {}
        for condition in design["condition_templates"]:
            key = (condition["topology"], condition["perturbation"])
            if key not in networks:
                networks[key] = build_v9_network(directory, *key)
        pending = []
        for seed, condition in task_specs:
            state_id = _state_id(seed, condition)
            if state_id in completed:
                continue
            network, metadata = networks[(condition["topology"], condition["perturbation"])]
            pending.append({
                "network": str(network), "metadata": metadata, "config": design,
                "condition": condition, "seed": seed,
                "analytical": analytical[f"{condition['id']}::{seed}"],
                "library": [action.to_dict() for action in library],
            })
        if pending:
            with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
                futures = {executor.submit(_execute_state, task): task for task in pending}
                for future in as_completed(futures):
                    result = future.result()
                    completed[result["state_id"]] = result
                    print(f"v9 actionability {len(completed)}/{len(task_specs)} {result['state_id']}", flush=True)
                    if len(completed) % 5 == 0:
                        _write(CHECKPOINT, sorted(completed.values(), key=lambda row: row["state_id"]))
    if len(completed) != len(task_specs):
        raise RuntimeError("v9 development actionability dataset is incomplete")
    states = sorted(completed.values(), key=lambda row: row["state_id"])
    rows = _action_rows(states, design)
    oracle = _oracle_report(rows)
    raw_path = OUTPUT / "raw_metrics.json"
    suffix = f"_limit_{limit}" if limit is not None else ""
    if suffix:
        raw_path = OUTPUT / f"raw_metrics{suffix}.json"
    _write(raw_path, rows)
    _write(OUTPUT / f"oracle_actionability{suffix}.json", oracle)
    _write(OUTPUT / f"dataset_summary{suffix}.json", {
        "complete": limit is None,
        "state_count": len(states),
        "state_action_row_count": len(rows),
        "actual_sumo_run_count": sum(2 + (24 if state["exhaustive_oracle"] else 8) for state in states),
        "exhaustive_state_count": sum(state["exhaustive_oracle"] for state in states),
        "role_state_counts": dict(Counter(_roles(list(design["development_seeds"]), design)[int(state["seed"])] for state in states)),
        "action_feature_count": len(ACTION_FEATURE_SCHEMA),
        "state_action_feature_count": len(STATE_ACTION_FEATURE_SCHEMA),
        "pairing_failure_count": sum(not all(row["pairing"].values()) for row in rows),
        "oracle_actionability_rate": oracle["all_evaluated_actionability"],
        "gate_A_pass": oracle["gate_A_pass"],
        "raw_metrics_hash": _sha(raw_path),
        "final_holdout_materialized": False,
        "rl_used": False,
    })
    if limit is None:
        CHECKPOINT.unlink(missing_ok=True)
    print(raw_path)
    return raw_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int)
    arguments = parser.parse_args()
    run(workers=arguments.workers, force=arguments.force, limit=arguments.limit)


if __name__ == "__main__":
    main()
