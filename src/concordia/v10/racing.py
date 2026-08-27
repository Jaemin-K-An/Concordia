from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

from .seeds import assert_seed_contract, racing_seed
from .statistics import empirical_lcb, mean_std, robust_mean


@dataclass(frozen=True)
class RolloutRequest:
    state_id: str
    action_id: str
    stage: str
    horizon_seconds: int
    replica: int
    seed: int


@dataclass(frozen=True)
class RolloutResult:
    state_id: str
    action_id: str
    stage: str
    horizon_seconds: int
    replica: int
    seed: int
    traffic_gain: float
    queue_delta: float
    risk_delta: float
    bottleneck_load_delta: float
    maximum_regret: float
    legal: bool

    @property
    def unsafe(self) -> bool:
        return bool(
            self.risk_delta > 0.25
            or self.maximum_regret > 0.08
            or not self.legal
        )

    def to_dict(self) -> dict:
        return {**asdict(self), "unsafe": self.unsafe}


BatchEvaluator = Callable[[Sequence[RolloutRequest]], Sequence[RolloutResult]]


def static_filter_reason(action: Mapping[str, object], plan: Mapping[str, object]) -> str | None:
    if bool(action.get("is_null", False)):
        return "null_fallback_not_raced"
    if not bool(plan.get("route_mapping_valid", True)):
        return "invalid_route_mapping"
    if not bool(plan.get("all_routes_legal", True)):
        return "illegal_route"
    if not bool(plan.get("preference_feasible", True)):
        return "preference_slack_violation"
    if float(plan.get("expected_accepted_user_count", 0.0)) <= 0.0:
        return "zero_expected_accepted_mass"
    if float(plan.get("destination_capacity_slack", 0.0)) < 0.0:
        return "structurally_impossible_target_capacity"
    return None


class MultiFidelityRacer:
    def __init__(self, config: Mapping[str, object]):
        self.config = dict(config)

    def _requests(
        self,
        state_id: str,
        action_ids: Sequence[str],
        stage: str,
        horizon: int,
        replicas: int,
        used_seeds: set[int],
    ) -> list[RolloutRequest]:
        requests = []
        for action_id in action_ids:
            for replica in range(replicas):
                seed = racing_seed(state_id, action_id, stage, replica)
                while seed in used_seeds:
                    seed = 100000 + (seed - 100000 + 1) % 900000
                used_seeds.add(seed)
                requests.append(RolloutRequest(
                    state_id, action_id, stage, horizon, replica, seed
                ))
        return requests

    @staticmethod
    def _group(results: Sequence[RolloutResult]) -> dict[str, list[RolloutResult]]:
        grouped: dict[str, list[RolloutResult]] = defaultdict(list)
        for result in results:
            grouped[result.action_id].append(result)
        return grouped

    def race(
        self,
        state_id: str,
        actions: Sequence[Mapping[str, object]],
        plans: Mapping[str, Mapping[str, object]],
        evaluator: BatchEvaluator,
        *,
        evaluation_seed: int,
    ) -> dict:
        action_map = {str(action["action_id"]): dict(action) for action in actions}
        if "A00_NULL_B1" not in action_map:
            raise ValueError("B1 null action must always exist")
        eliminated: dict[str, dict] = {}
        active = []
        for action_id, action in action_map.items():
            if action_id == "A00_NULL_B1":
                continue
            reason = static_filter_reason(action, plans[action_id])
            if reason:
                eliminated[action_id] = {"stage": "stage_0", "reason": reason}
            else:
                active.append(action_id)
        stage0 = tuple(sorted(active))
        all_requests: list[RolloutRequest] = []
        all_results: list[RolloutResult] = []
        used_seeds: set[int] = set()

        stage1_config = self.config["stage_1"]
        stage1_requests = self._requests(
            state_id, active, "stage_1",
            int(stage1_config["horizon_seconds"]), int(stage1_config["replicas"]),
            used_seeds,
        )
        stage1_results = list(evaluator(stage1_requests))
        all_requests.extend(stage1_requests)
        all_results.extend(stage1_results)
        fixed_uncertainty = float(stage1_config["fixed_single_rollout_uncertainty_half_width"])
        eta = float(stage1_config["conservative_elimination_eta"])
        kill = float(stage1_config["safety_kill_risk_delta"])
        scored_stage1 = []
        for action_id, values in self._group(stage1_results).items():
            value = values[0]
            if value.risk_delta > kill or value.maximum_regret > 0.08 or not value.legal:
                eliminated[action_id] = {"stage": "stage_1", "reason": "obvious_safety_or_regret_failure"}
                continue
            composite = (
                value.traffic_gain
                - 0.10 * max(0.0, value.risk_delta)
                - 0.01 * max(0.0, value.queue_delta)
                - 0.01 * max(0.0, value.bottleneck_load_delta)
            )
            scored_stage1.append((action_id, composite, composite - fixed_uncertainty, composite + fixed_uncertainty))
        best_lcb = max((value[2] for value in scored_stage1), default=-np.inf)
        plausible = [value for value in scored_stage1 if value[3] >= best_lcb - eta]
        plausible.sort(key=lambda value: (-value[2], -value[1], value[0]))
        target1 = int(stage1_config["target_survivors"])
        minimum1 = min(int(stage1_config["minimum_survivors"]), len(scored_stage1))
        if len(plausible) < minimum1:
            ranked = sorted(scored_stage1, key=lambda value: (-value[2], -value[1], value[0]))
            plausible = ranked[:minimum1]
        survivors1 = [value[0] for value in plausible[:target1]]
        for action_id, *_values in scored_stage1:
            if action_id not in survivors1:
                eliminated[action_id] = {"stage": "stage_1", "reason": "conservative_racing_elimination"}

        stage2_config = self.config["stage_2"]
        stage2_requests = self._requests(
            state_id, survivors1, "stage_2",
            int(stage2_config["horizon_seconds"]), int(stage2_config["replicas"]),
            used_seeds,
        )
        stage2_results = list(evaluator(stage2_requests))
        all_requests.extend(stage2_requests)
        all_results.extend(stage2_results)
        scored_stage2 = []
        for action_id, values in self._group(stage2_results).items():
            unsafe_count = sum(result.unsafe for result in values)
            if unsafe_count >= int(stage2_config["unsafe_count_elimination_threshold"]):
                eliminated[action_id] = {"stage": "stage_2", "reason": "recurring_unsafe_rollout"}
                continue
            score = robust_mean(
                [result.traffic_gain for result in values],
                float(stage2_config["initial_variance_penalty"]),
            )
            scored_stage2.append((action_id, score, unsafe_count))
        scored_stage2.sort(key=lambda value: (-value[1], value[2], value[0]))
        survivors2 = [value[0] for value in scored_stage2[: int(stage2_config["target_survivors"])]]
        for action_id, *_values in scored_stage2:
            if action_id not in survivors2:
                eliminated[action_id] = {"stage": "stage_2", "reason": "robust_score_elimination"}

        stage3_config = self.config["stage_3"]
        stage3_requests = self._requests(
            state_id, survivors2, "stage_3",
            int(stage3_config["horizon_seconds"]), int(stage3_config["replicas"]),
            used_seeds,
        )
        stage3_results = list(evaluator(stage3_requests))
        all_requests.extend(stage3_requests)
        all_results.extend(stage3_results)
        eligible = []
        stage3_statistics = {}
        for action_id, values in self._group(stage3_results).items():
            traffic = [result.traffic_gain for result in values]
            mean, standard_deviation = mean_std(traffic)
            lcb = empirical_lcb(
                traffic, float(stage3_config.get("lcb_quantile", 0.10))
            )
            unsafe_count = sum(result.unsafe for result in values)
            statistics = {
                "mean_traffic_gain": mean,
                "traffic_gain_std": standard_deviation,
                "q10_lcb": lcb,
                "maximum_risk_delta": max(result.risk_delta for result in values),
                "unsafe_count": unsafe_count,
                "maximum_regret": max(result.maximum_regret for result in values),
                "all_legal": all(result.legal for result in values),
            }
            stage3_statistics[action_id] = statistics
            passes = bool(
                lcb > float(stage3_config["minimum_lcb_relative_benefit"])
                and unsafe_count <= int(stage3_config["maximum_unsafe_count"])
                and statistics["maximum_risk_delta"] <= float(stage3_config["maximum_replica_risk_delta"])
                and statistics["maximum_regret"] <= float(stage3_config["maximum_regret"])
                and statistics["all_legal"]
            )
            if passes:
                eligible.append((action_id, lcb))
            else:
                eliminated[action_id] = {"stage": "stage_3", "reason": "final_eligibility_failure"}
        eligible.sort(key=lambda value: (-value[1], value[0]))
        finalists = [value[0] for value in eligible[: int(stage3_config["target_survivors"])]]
        for action_id, _score in eligible:
            if action_id not in finalists:
                eliminated[action_id] = {"stage": "stage_3", "reason": "top3_lcb_elimination"}

        verification_config = self.config["verification"]
        verification_requests = self._requests(
            state_id, finalists, "verification",
            int(stage3_config["horizon_seconds"]), int(verification_config["fresh_replicas"]),
            used_seeds,
        )
        verification_results = list(evaluator(verification_requests))
        all_requests.extend(verification_requests)
        all_results.extend(verification_results)
        verified = []
        verification_statistics = {}
        for action_id, values in self._group(verification_results).items():
            traffic = [result.traffic_gain for result in values]
            mean, _standard_deviation = mean_std(traffic)
            minimum = min(traffic)
            unsafe_count = sum(result.unsafe for result in values)
            verification_statistics[action_id] = {
                "mean_traffic_gain": mean,
                "minimum_traffic_gain": minimum,
                "unsafe_count": unsafe_count,
            }
            if (
                unsafe_count <= int(verification_config["maximum_unsafe_count"])
                and mean > float(verification_config["minimum_mean_relative_benefit_strict"])
                and minimum
                > float(verification_config.get("minimum_replica_relative_benefit", -1.0))
            ):
                verified.append(action_id)
            else:
                eliminated[action_id] = {"stage": "verification", "reason": "fresh_verification_failure"}
        selected = max(
            verified,
            key=lambda action_id: (stage3_statistics[action_id]["q10_lcb"], action_id),
            default="A00_NULL_B1",
        )
        assert_seed_contract([request.seed for request in all_requests], [evaluation_seed])
        return {
            "state_id": state_id,
            "selected_action_id": selected,
            "intervene": selected != "A00_NULL_B1",
            "stage_0_survivors": list(stage0),
            "stage_1_survivors": survivors1,
            "stage_2_survivors": survivors2,
            "stage_3_survivors": finalists,
            "verified_actions": sorted(verified),
            "eliminated": eliminated,
            "stage_3_statistics": stage3_statistics,
            "verification_statistics": verification_statistics,
            "rollout_requests": [asdict(request) for request in all_requests],
            "rollout_results": [result.to_dict() for result in all_results],
            "decision_evaluation_seed_overlap": 0,
        }
