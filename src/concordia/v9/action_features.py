from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from concordia.uplift_v7.paired_dataset import UPLIFT_V7_FEATURE_SCHEMA

from .action_space import ROUTE_ALLOCATIONS, USER_STRATEGIES


ACTION_NUMERIC_FEATURES = (
    "reroute_fraction",
    "proposed_rerouted_user_count",
    "expected_accepted_user_count",
    "expected_rerouted_flow",
    "maximum_target_edge_load_increase",
    "maximum_source_edge_load_reduction",
    "bottleneck_flow_delta",
    "destination_capacity_slack",
    "route_overlap_delta",
    "route_entropy",
    "affected_junction_count",
    "expected_lane_change_demand_delta",
    "conflict_zone_exposure_delta",
    "action_concentration_index",
    "average_preference_slack_selected",
    "p90_preference_slack_selected",
    "expected_acceptance_probability",
    "route_length_increase",
    "reliability_change",
    "is_null_action",
)
ACTION_ONE_HOT_FEATURES = (
    *(f"strategy_{name}" for name in USER_STRATEGIES),
    *(f"allocation_{name}" for name in ROUTE_ALLOCATIONS),
)
ACTION_FEATURE_SCHEMA = (*ACTION_NUMERIC_FEATURES, *ACTION_ONE_HOT_FEATURES)
STATE_ACTION_FEATURE_SCHEMA = (*UPLIFT_V7_FEATURE_SCHEMA, *ACTION_FEATURE_SCHEMA)
FORBIDDEN_TOKENS = ("realized", "outcome", "tau_", "risk_adaptive", "ttt_adaptive", "actual_acceptance")


def build_action_features(action: Mapping[str, object], plan: Mapping[str, object]) -> dict[str, float]:
    output = {
        name: float(plan.get(name, 0.0))
        for name in ACTION_NUMERIC_FEATURES
    }
    output["reroute_fraction"] = float(action["reroute_fraction"])
    output["is_null_action"] = float(bool(action.get("is_null", False)))
    for strategy in USER_STRATEGIES:
        output[f"strategy_{strategy}"] = float(str(action["user_strategy"]) == strategy)
    for allocation in ROUTE_ALLOCATIONS:
        output[f"allocation_{allocation}"] = float(str(action["route_allocation"]) == allocation)
    validate_action_features(output)
    return {name: output[name] for name in ACTION_FEATURE_SCHEMA}


def validate_action_features(features: Mapping[str, object]) -> None:
    if set(features) != set(ACTION_FEATURE_SCHEMA):
        raise ValueError("v9 action feature schema mismatch")
    for name, value in features.items():
        if any(token in name.lower() for token in FORBIDDEN_TOKENS):
            raise ValueError(f"future leakage in v9 action feature: {name}")
        if not math.isfinite(float(value)):
            raise ValueError(f"non-finite v9 action feature: {name}")


def state_action_features(
    state_features: Mapping[str, object], action_features: Mapping[str, object]
) -> dict[str, float]:
    output = {name: float(state_features[name]) for name in UPLIFT_V7_FEATURE_SCHEMA}
    output.update({name: float(action_features[name]) for name in ACTION_FEATURE_SCHEMA})
    if set(output) != set(STATE_ACTION_FEATURE_SCHEMA):
        raise ValueError("v9 state-action schema mismatch")
    return output


def feature_matrix(
    rows: Sequence[Mapping[str, object]],
    feature_names: Sequence[str] = STATE_ACTION_FEATURE_SCHEMA,
) -> np.ndarray:
    return np.asarray(
        [[float(row["state_action_features"][name]) for name in feature_names] for row in rows],
        dtype=float,
    )

