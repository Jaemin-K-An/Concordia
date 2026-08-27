#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import fields
from pathlib import Path
from typing import Mapping, Sequence

import yaml

from concordia.v10.cache import RolloutCache
from concordia.v10.racing import MultiFidelityRacer, RolloutRequest, RolloutResult
from concordia.v9.action_space import AdaptiveAction, b6_reference_action, generate_action_library
from v10_micro_sim import build_v9_network, paired_rollout_result, run_v10_candidate


ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "artifacts/cache/v10_rollouts"
STUDY_DIR = ROOT / "artifacts/studies/v10_racing_validation"
ROLLOUT_FIELDS = {field.name for field in fields(RolloutResult)}


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _hash_value(value: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _action_from_value(value: Mapping[str, object]):
    if value.get("reference_only"):
        return dict(value)
    return AdaptiveAction.from_dict(value)


def _execute_candidate(task: Mapping[str, object]) -> dict:
    return run_v10_candidate(
        Path(task["network"]), task["metadata"], task["simulation_config"],
        task["condition"], int(task["state_seed"]), int(task["future_seed"]),
        _action_from_value(task["action"]), int(task["horizon_seconds"]),
        plan_actions=[_action_from_value(value) for value in task.get("plan_actions", [])],
        demand_multiplier=float(task.get("demand_multiplier", 1.0)),
        reaction_multiplier=float(task.get("reaction_multiplier", 1.0)),
    )


class StateRolloutEvaluator:
    def __init__(
        self,
        *,
        state_id: str,
        state_seed: int,
        network: Path,
        metadata: Mapping[str, object],
        simulation_config: Mapping[str, object],
        condition: Mapping[str, object],
        actions: Sequence[Mapping[str, object]],
        executor: ProcessPoolExecutor,
        parameter_label: str = "decision_nominal",
    ):
        self.state_id = state_id
        self.state_seed = state_seed
        self.network = network
        self.metadata = dict(metadata)
        self.simulation_config = dict(simulation_config)
        self.condition = dict(condition)
        self.actions = {str(action["action_id"]): dict(action) for action in actions}
        self.executor = executor
        self.cache = RolloutCache(CACHE_DIR)
        self.baselines: dict[int, dict] = {}
        self.normalized_network_hash_pair_count = 0
        self.parameter_hash = _hash_value({
            "parameter_label": parameter_label,
            "condition": self.condition,
            "state_seed": state_seed,
            "network_metadata": self.metadata,
            "simulator_source_hash": hashlib.sha256(
                (ROOT / "scripts/v10_micro_sim.py").read_bytes()
            ).hexdigest(),
        })

    def _task(
        self,
        action_id: str,
        future_seed: int,
        horizon_seconds: int,
        *,
        plan_actions: Sequence[Mapping[str, object]] = (),
        demand_multiplier: float = 1.0,
        reaction_multiplier: float = 1.0,
    ) -> dict:
        return {
            "network": str(self.network),
            "metadata": self.metadata,
            "simulation_config": self.simulation_config,
            "condition": self.condition,
            "state_seed": self.state_seed,
            "future_seed": future_seed,
            "action": self.actions[action_id],
            "horizon_seconds": horizon_seconds,
            "plan_actions": list(plan_actions),
            "demand_multiplier": demand_multiplier,
            "reaction_multiplier": reaction_multiplier,
        }

    def baseline(
        self,
        horizon_seconds: int,
        *,
        plan_actions: Sequence[Mapping[str, object]] = (),
    ) -> dict:
        if horizon_seconds not in self.baselines:
            cache_key = {
                "state_id": self.state_id,
                "action_id": "A00_NULL_B1",
                "stage": f"baseline_{horizon_seconds}",
                "replica": 0,
                "horizon_seconds": horizon_seconds,
                "simulator_parameter_hash": self.parameter_hash,
            }
            cached = self.cache.load(cache_key)
            if cached is not None and (
                not plan_actions or bool(cached.get("candidate_plans"))
            ):
                self.baselines[horizon_seconds] = cached
            else:
                baseline = _execute_candidate(self._task(
                    "A00_NULL_B1", self.state_seed, horizon_seconds,
                    plan_actions=plan_actions,
                ))
                self.cache.store(cache_key, baseline)
                self.baselines[horizon_seconds] = baseline
        return self.baselines[horizon_seconds]

    def __call__(self, requests: Sequence[RolloutRequest]) -> Sequence[RolloutResult]:
        if not requests:
            return []
        horizons = sorted({request.horizon_seconds for request in requests})
        for horizon in horizons:
            self.baseline(horizon)
        output: dict[tuple[str, str, int], RolloutResult] = {}
        pending = {}
        for request in requests:
            cache_key = {
                "state_id": request.state_id,
                "action_id": request.action_id,
                "stage": request.stage,
                "replica": request.replica,
                "horizon_seconds": request.horizon_seconds,
                "simulator_parameter_hash": self.parameter_hash,
            }
            cached = self.cache.load(cache_key)
            identity = (request.action_id, request.stage, request.replica)
            if cached is not None and int(cached["seed"]) == request.seed:
                output[identity] = RolloutResult(**{
                    name: cached[name] for name in ROLLOUT_FIELDS
                })
                continue
            future = self.executor.submit(
                _execute_candidate,
                self._task(
                    request.action_id, request.seed, request.horizon_seconds
                ),
            )
            pending[future] = (request, cache_key, identity)
        for future in as_completed(pending):
            request, cache_key, identity = pending[future]
            action = future.result()
            baseline = dict(self.baseline(request.horizon_seconds))
            baseline["future_seed"] = request.seed
            if (
                baseline.get("network_hash") != action.get("network_hash")
                and baseline.get("route_file_hash") == action.get("route_file_hash")
            ):
                # netconvert embeds generation timestamps and temporary input paths
                # in an XML comment. The registered topology, perturbation, metadata,
                # route bytes, and simulator parameter hash are already identical in
                # this evaluator/cache namespace, so normalize only that non-semantic
                # byte hash before applying the paired contract.
                baseline["network_hash"] = action["network_hash"]
                self.normalized_network_hash_pair_count += 1
            try:
                paired = paired_rollout_result(baseline, action)
            except RuntimeError as error:
                contract_fields = (
                    "state_seed", "future_seed", "network_hash", "route_file_hash"
                )
                differences = {
                    name: {"baseline": baseline.get(name), "action": action.get(name)}
                    for name in contract_fields
                    if baseline.get(name) != action.get(name)
                }
                raise RuntimeError(
                    f"v10 paired rollout contract failed: {differences}"
                ) from error
            result = RolloutResult(
                request.state_id, request.action_id, request.stage,
                request.horizon_seconds, request.replica, request.seed,
                **paired,
            )
            output[identity] = result
            self.cache.store(cache_key, result.to_dict())
        return [output[(request.action_id, request.stage, request.replica)] for request in requests]

    def actual_actions(
        self,
        action_ids: Sequence[str],
        *,
        horizon_seconds: int,
        evaluation_seed: int,
        demand_multiplier: float = 1.0,
        reaction_multiplier: float = 1.0,
    ) -> tuple[dict, dict[str, dict]]:
        baseline_future = self.executor.submit(
            _execute_candidate,
            self._task(
                "A00_NULL_B1", evaluation_seed, horizon_seconds,
                demand_multiplier=demand_multiplier,
                reaction_multiplier=reaction_multiplier,
            ),
        )
        futures = {
            self.executor.submit(
                _execute_candidate,
                self._task(
                    action_id, evaluation_seed, horizon_seconds,
                    demand_multiplier=demand_multiplier,
                    reaction_multiplier=reaction_multiplier,
                ),
            ): action_id
            for action_id in action_ids
            if action_id != "A00_NULL_B1"
        }
        baseline = baseline_future.result()
        outcomes = {
            "A00_NULL_B1": {
                "traffic_gain": 0.0, "queue_delta": 0.0, "risk_delta": 0.0,
                "bottleneck_load_delta": 0.0, "maximum_regret": 0.0,
                "legal": True,
            }
        }
        for future in as_completed(futures):
            action_id = futures[future]
            action = future.result()
            outcomes[action_id] = paired_rollout_result(baseline, action)
            outcomes[action_id].update({
                "ttt_b1": float(baseline["completed_travel_time_seconds"]),
                "ttt_action": float(action["completed_travel_time_seconds"]),
                "generated_b1": int(baseline["generated_vehicle_count"]),
                "arrived_b1": int(baseline["arrived_vehicle_count"]),
                "generated_action": int(action["generated_vehicle_count"]),
                "arrived_action": int(action["arrived_vehicle_count"]),
            })
            if (
                outcomes[action_id]["arrived_b1"] == outcomes[action_id]["generated_b1"]
                and outcomes[action_id]["arrived_action"] == outcomes[action_id]["generated_action"]
            ):
                outcomes[action_id]["traffic_gain"] = (
                    outcomes[action_id]["ttt_b1"] - outcomes[action_id]["ttt_action"]
                ) / max(outcomes[action_id]["ttt_b1"], 1e-9)
        return baseline, outcomes


def _state_specs(partition: str, limit: int | None) -> tuple[dict, list[tuple[int, dict]]]:
    design = yaml.safe_load((ROOT / f"configs/v10/{partition}_design.yaml").read_text())
    source = yaml.safe_load((ROOT / design["condition_templates_source"]).read_text())
    specs = [
        (int(seed), dict(condition))
        for seed in design["seed_families"]
        for condition in source["condition_templates"]
    ]
    return design, specs[:limit] if limit is not None else specs


def _state_id(partition: str, seed: int, condition: Mapping[str, object]) -> str:
    prefix = "V10D" if partition == "development" else "V10V"
    return f"{prefix}::{condition['id']}::s{seed}"


def _safe_success(outcome: Mapping[str, object]) -> bool:
    return bool(
        float(outcome["traffic_gain"]) > 0.005
        and float(outcome["risk_delta"]) <= 0.25
        and float(outcome["maximum_regret"]) <= 0.08
        and bool(outcome["legal"])
    )


def _mismatch(state_id: str) -> tuple[float, float]:
    digest = int(hashlib.sha256(state_id.encode()).hexdigest()[:8], 16)
    demand = 0.97 if digest % 2 == 0 else 1.03
    reaction = 0.95 if (digest // 2) % 2 == 0 else 1.05
    return demand, reaction


def _development_record(
    *,
    trace: dict,
    outcomes: Mapping[str, Mapping[str, object]],
    actions: Sequence[Mapping[str, object]],
) -> dict:
    raced_action_ids = {str(action["action_id"]) for action in actions}
    feasible = {
        action_id: outcome for action_id, outcome in outcomes.items()
        if action_id in raced_action_ids and (
            float(outcome["risk_delta"]) <= 0.25
            and float(outcome["maximum_regret"]) <= 0.08
            and bool(outcome["legal"])
        )
    }
    oracle_action = max(
        feasible, key=lambda action_id: (float(feasible[action_id]["traffic_gain"]), action_id)
    )
    oracle_beneficial = _safe_success(feasible[oracle_action])
    selected = trace["selected_action_id"]
    selected_outcome = outcomes[selected]
    stages = {
        "stage_1": trace["stage_1_survivors"],
        "stage_2": trace["stage_2_survivors"],
        "stage_3": trace["stage_3_survivors"],
        "verification": trace["verified_actions"],
    }
    survival = {
        name: bool(not oracle_beneficial or oracle_action in survivors)
        for name, survivors in stages.items()
    }
    if not oracle_beneficial:
        failure = "F1_no_beneficial_action"
    elif not survival["stage_1"]:
        failure = "F2_good_action_eliminated_stage1"
    elif not survival["stage_2"]:
        failure = "F3_good_action_eliminated_stage2"
    elif not survival["stage_3"]:
        failure = "F4_good_action_eliminated_stage3"
    elif selected != "A00_NULL_B1" and float(selected_outcome["risk_delta"]) > 0.25:
        failure = "F6_safety_mismatch"
    elif selected != "A00_NULL_B1" and float(selected_outcome["maximum_regret"]) > 0.08:
        failure = "F8_regret_acceptance_mismatch"
    elif selected != "A00_NULL_B1" and not _safe_success(selected_outcome):
        failure = "F5_digital_twin_false_positive"
    elif selected != oracle_action:
        failure = "F7_stochastic_reversal"
    else:
        failure = None
    b6_id = b6_reference_action()["action_id"]
    return {
        "trace": trace,
        "oracle_action_id": oracle_action,
        "oracle_beneficial": oracle_beneficial,
        "oracle_benefit": float(feasible[oracle_action]["traffic_gain"]),
        "oracle_survival": survival,
        "selected_outcome": selected_outcome,
        "selected_safe_success": _safe_success(selected_outcome) if trace["intervene"] else False,
        "selection_regret": max(
            0.0,
            float(feasible[oracle_action]["traffic_gain"])
            - float(selected_outcome["traffic_gain"]),
        ),
        "action_outcomes": dict(outcomes),
        "b6_outcome": outcomes[b6_id],
        "failure_taxonomy": failure,
        "action_count": len(actions) - 1,
    }


def _validation_record(trace: dict, outcome: Mapping[str, object], mismatch: tuple[float, float]) -> dict:
    return {
        "trace": trace,
        "selected_outcome": dict(outcome),
        "selected_safe_success": _safe_success(outcome) if trace["intervene"] else False,
        "mismatch": {"demand_multiplier": mismatch[0], "reaction_multiplier": mismatch[1]},
    }


def _summary(partition: str, records: Sequence[Mapping[str, object]]) -> dict:
    interventions = [record for record in records if record["trace"]["intervene"]]
    successes = sum(bool(record["selected_safe_success"]) for record in interventions)
    safety_violations = sum(
        float(record["selected_outcome"]["risk_delta"]) > 0.25
        for record in interventions
    )
    regret_violations = sum(
        float(record["selected_outcome"]["maximum_regret"]) > 0.08
        for record in interventions
    )
    legal_violations = sum(
        not bool(record["selected_outcome"]["legal"])
        for record in interventions
    )
    summary = {
        "partition": partition,
        "state_count": len(records),
        "intervention_count": len(interventions),
        "coverage": len(interventions) / len(records) if records else 0.0,
        "safe_beneficial_intervention_count": successes,
        "precision": successes / len(interventions) if interventions else 0.0,
        "safety_violation_count": safety_violations,
        "regret_violation_count": regret_violations,
        "legal_violation_count": legal_violations,
        "any_constraint_violation_count": sum(
            float(record["selected_outcome"]["risk_delta"]) > 0.25
            or float(record["selected_outcome"]["maximum_regret"]) > 0.08
            or not bool(record["selected_outcome"]["legal"])
            for record in interventions
        ),
        "decision_evaluation_seed_overlap": sum(
            int(record["trace"]["decision_evaluation_seed_overlap"]) for record in records
        ),
        "rollout_count": sum(len(record["trace"]["rollout_results"]) for record in records),
        "final_holdout_materialized": False,
        "rl_used": False,
    }
    latencies = sorted(
        float(record["trace"]["decision_latency_seconds"])
        for record in records
        if "decision_latency_seconds" in record["trace"]
    )
    if latencies:
        summary["mean_decision_latency_seconds"] = sum(latencies) / len(latencies)
        summary["p95_decision_latency_seconds"] = latencies[
            min(len(latencies) - 1, int(0.95 * len(latencies)))
        ]
    if partition == "development":
        actionable = [record for record in records if record["oracle_beneficial"]]
        summary.update({
            "oracle_actionable_state_count": len(actionable),
            "stage_1_oracle_survival": sum(record["oracle_survival"]["stage_1"] for record in actionable) / len(actionable) if actionable else 0.0,
            "stage_2_oracle_survival": sum(record["oracle_survival"]["stage_2"] for record in actionable) / len(actionable) if actionable else 0.0,
            "stage_3_oracle_survival": sum(record["oracle_survival"]["stage_3"] for record in actionable) / len(actionable) if actionable else 0.0,
            "verification_oracle_survival": sum(record["oracle_survival"]["verification"] for record in actionable) / len(actionable) if actionable else 0.0,
            "mean_selection_regret": sum(float(record["selection_regret"]) for record in records) / len(records) if records else 0.0,
        })
    else:
        summary["authorization"] = {
            "precision_at_least_0_85": summary["precision"] >= 0.85,
            "safety_violations_zero": safety_violations == 0,
            "interventions_at_least_30": len(interventions) >= 30,
            "coverage_at_least_0_10": summary["coverage"] >= 0.10,
        }
        summary["freeze_authorized"] = all(summary["authorization"].values())
    return summary


def run(
    partition: str,
    *,
    workers: int,
    limit: int | None = None,
    force: bool = False,
    racing_config_path: Path | None = None,
    evidence_label: str | None = None,
    reuse_development_outcomes: Path | None = None,
) -> Path:
    if partition not in {"development", "validation"}:
        raise ValueError("partition must be development or validation")
    if (ROOT / "artifacts/studies/v10_micro_holdout/summary.json").exists():
        raise RuntimeError("v10 development/validation cannot run after final materialization")
    design, specs = _state_specs(partition, limit)
    simulation_config = yaml.safe_load((ROOT / "configs/v9/development_design.yaml").read_text())
    racing_config_path = racing_config_path or ROOT / "configs/v10/racing_design.yaml"
    racing_config = yaml.safe_load(racing_config_path.read_text())
    racer = MultiFidelityRacer(racing_config)
    actions = [action.to_dict() for action in generate_action_library()]
    actions_with_b6 = [*actions, b6_reference_action()]
    evidence_label = evidence_label or partition
    checkpoint = STUDY_DIR / f"{evidence_label}_checkpoint.json"
    reused_outcomes = {}
    if reuse_development_outcomes is not None:
        if partition != "development":
            raise ValueError("realized outcomes may only be reused for development repairs")
        reused_outcomes = {
            record["state_id"]: record["action_outcomes"]
            for record in json.loads(reuse_development_outcomes.read_text())
        }
    completed = {
        record["state_id"]: record
        for record in (
            json.loads(checkpoint.read_text())
            if checkpoint.is_file() and not force else []
        )
    }
    with tempfile.TemporaryDirectory(prefix=f"concordia-v10-{partition}-networks-") as temporary:
        directory = Path(temporary)
        networks = {}
        for _seed, condition in specs:
            key = (condition["topology"], condition["perturbation"])
            if key not in networks:
                networks[key] = build_v9_network(directory, *key)
        with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
            for seed, condition in specs:
                state_id = _state_id(partition, seed, condition)
                if state_id in completed:
                    continue
                network, metadata = networks[(condition["topology"], condition["perturbation"])]
                evaluator = StateRolloutEvaluator(
                    state_id=state_id, state_seed=seed, network=network,
                    metadata=metadata, simulation_config=simulation_config,
                    condition=condition, actions=actions_with_b6, executor=executor,
                )
                baseline = evaluator.baseline(
                    int(racing_config["stage_1"]["horizon_seconds"]),
                    plan_actions=actions,
                )
                plans = baseline["candidate_plans"]
                decision_started = time.perf_counter()
                trace = racer.race(
                    state_id, actions, plans, evaluator, evaluation_seed=seed
                )
                trace["decision_latency_seconds"] = time.perf_counter() - decision_started
                trace["normalized_network_hash_pair_count"] = (
                    evaluator.normalized_network_hash_pair_count
                )
                if partition == "development":
                    outcomes = reused_outcomes.get(state_id)
                    if outcomes is None:
                        _baseline, outcomes = evaluator.actual_actions(
                            [action["action_id"] for action in actions_with_b6],
                            horizon_seconds=300, evaluation_seed=seed,
                        )
                    detail = _development_record(
                        trace=trace, outcomes=outcomes, actions=actions
                    )
                else:
                    mismatch = _mismatch(state_id)
                    selected = trace["selected_action_id"]
                    if selected == "A00_NULL_B1":
                        outcome = {
                            "traffic_gain": 0.0, "queue_delta": 0.0,
                            "risk_delta": 0.0, "bottleneck_load_delta": 0.0,
                            "maximum_regret": 0.0, "legal": True,
                        }
                    else:
                        _baseline, outcomes = evaluator.actual_actions(
                            [selected], horizon_seconds=300, evaluation_seed=seed,
                            demand_multiplier=mismatch[0],
                            reaction_multiplier=mismatch[1],
                        )
                        outcome = outcomes[selected]
                    detail = _validation_record(trace, outcome, mismatch)
                completed[state_id] = {
                    "state_id": state_id,
                    "seed": seed,
                    "template_id": condition["id"],
                    "condition": condition,
                    **detail,
                }
                print(
                    f"v10 {partition} {len(completed)}/{len(specs)} {state_id} "
                    f"selected={trace['selected_action_id']}",
                    flush=True,
                )
                _write(checkpoint, sorted(completed.values(), key=lambda value: value["state_id"]))
    records = [completed[_state_id(partition, seed, condition)] for seed, condition in specs]
    suffix = f"_limit_{limit}" if limit is not None else ""
    raw_path = STUDY_DIR / f"{evidence_label}_raw_metrics{suffix}.json"
    summary_path = STUDY_DIR / f"{evidence_label}_summary{suffix}.json"
    _write(raw_path, records)
    summary = _summary(partition, records)
    summary.update({
        "evidence_label": evidence_label,
        "racing_config": str(racing_config_path.resolve().relative_to(ROOT)),
        "racing_config_sha256": hashlib.sha256(racing_config_path.read_bytes()).hexdigest(),
        "normalized_network_hash_pair_count": sum(
            int(record["trace"].get("normalized_network_hash_pair_count", 0))
            for record in records
        ),
    })
    _write(summary_path, summary)
    if limit is None:
        checkpoint.unlink(missing_ok=True)
    print(summary_path)
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("partition", choices=("development", "validation"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--racing-config", type=Path)
    parser.add_argument("--evidence-label")
    parser.add_argument("--reuse-development-outcomes", type=Path)
    arguments = parser.parse_args()
    run(
        arguments.partition,
        workers=arguments.workers,
        limit=arguments.limit,
        force=arguments.force,
        racing_config_path=arguments.racing_config,
        evidence_label=arguments.evidence_label,
        reuse_development_outcomes=arguments.reuse_development_outcomes,
    )


if __name__ == "__main__":
    main()
