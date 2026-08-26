from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from concordia.uplift_v7.paired_dataset import UPLIFT_V7_FEATURE_SCHEMA


SAFETY_PRECURSOR_FEATURES = (
    "headway_p05_proxy",
    "headway_p10_proxy",
    "closing_speed_p95_proxy",
    "ttc_lower_proxy",
    "drac_p90_proxy",
    "traffic_instability_proxy",
    "entering_flow_ratio",
    "conflicting_movement_count",
    "signal_phase_pressure",
)
STATE_INTERACTION_FEATURES = (
    "closing_speed_x_min_headway",
    "vc_x_speed_variance",
    "penetration_x_lane_change_density",
    "merge_density_x_flow_instability",
)
CANDIDATE_RANK_FEATURES = (
    "traffic_uplift_score",
    "traffic_rank_percentile",
    "traffic_uplift_x_drac_proxy",
)
ACTION_FEATURES = (
    "proposed_rerouted_user_count",
    "proposed_reroute_fraction",
    "proposed_rerouted_flow_vph",
    "destination_edge_load_change",
    "maximum_edge_flow_delta",
    "conflict_zone_exposure_delta",
    "lane_change_demand_delta",
    "bottleneck_load_delta",
    "route_overlap_delta",
    "capacity_slack_delta",
    "expected_acceptance_mass",
    "action_origin_merge",
    "action_origin_signalized",
    "action_origin_two_route",
    "action_origin_asymmetric",
    "action_origin_real_like",
    "action_destination_primary_alternative",
    "action_destination_secondary_alternative",
)

STATE_ONLY_FEATURE_SCHEMA = (
    *UPLIFT_V7_FEATURE_SCHEMA,
    *SAFETY_PRECURSOR_FEATURES,
    *STATE_INTERACTION_FEATURES,
)
ACTION_AWARE_FEATURE_SCHEMA = (
    *STATE_ONLY_FEATURE_SCHEMA,
    *CANDIDATE_RANK_FEATURES,
    *ACTION_FEATURES,
)

FORBIDDEN_INPUT_TOKENS = (
    "realized_acceptance",
    "post_treatment",
    "adaptive_ttt",
    "risk_adaptive",
    "tau_s",
    "unsafe_intervention",
)


def _clip(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def action_aware_features(
    row: Mapping[str, object],
    *,
    traffic_uplift_score: float,
    traffic_rank_percentile: float,
) -> dict[str, float]:
    """Construct only pre-decision state and proposed-action expectations."""
    base = {name: float(row["predecision_features"][name]) for name in UPLIFT_V7_FEATURE_SCHEMA}
    condition = dict(row["condition"])
    headway = max(base["minimum_headway"], 0.02)
    closing = max(base["closing_speed_p90"], 0.0)
    drac = max(base["drac_proxy_p95"], 0.0)
    demand = float(condition.get("demand", base["demand_vehicles_per_hour"]))
    penetration = _clip(base["navigation_penetration"], 0.0, 1.0)
    predicted_acceptance = _clip(
        base["predicted_acceptance"] * float(condition.get("acceptance_multiplier", 1.0)),
        0.0,
        1.0,
    )
    proposed_fraction = penetration * predicted_acceptance
    expected_vehicle_mass = demand * 120.0 / 3600.0
    expected_acceptance_mass = expected_vehicle_mass * proposed_fraction
    proposed_flow = demand * proposed_fraction
    capacity_ratio = max(base["alternative_capacity_ratio"], 0.10)
    vc = max(base["volume_capacity_ratio"], 0.0)
    topology = str(condition.get("topology", "unknown"))
    merge = float(topology == "merge")
    signalized = float(topology == "signalized")
    two_route = float(topology == "two_route")
    asymmetric = float(topology == "asymmetric")
    real_like = float(topology == "real_like")
    conflict_count = 1.0 + 2.0 * merge + 3.0 * signalized + asymmetric + real_like
    normalized_flow_delta = proposed_flow / max(demand * capacity_ratio, 1.0)

    features = {
        **base,
        "headway_p05_proxy": 0.82 * headway,
        "headway_p10_proxy": 0.90 * headway,
        "closing_speed_p95_proxy": 1.12 * closing,
        "ttc_lower_proxy": headway / max(closing, 0.10),
        "drac_p90_proxy": 0.90 * drac,
        "traffic_instability_proxy": abs(base["flow_instability"])
        + abs(base["speed_slope_30s"])
        + base["short_horizon_speed_oscillation"],
        "entering_flow_ratio": demand / max(base["flow_mean"], 1.0),
        "conflicting_movement_count": conflict_count,
        "signal_phase_pressure": signalized
        * (vc + base["queue_growth_rate"] + base["occupancy_mean"]),
        "closing_speed_x_min_headway": closing * headway,
        "vc_x_speed_variance": vc * base["speed_variance"],
        "penetration_x_lane_change_density": penetration * base["lane_change_density"],
        "merge_density_x_flow_instability": base["merge_interaction_density"]
        * base["flow_instability"],
        "traffic_uplift_score": float(traffic_uplift_score),
        "traffic_rank_percentile": _clip(float(traffic_rank_percentile), 0.0, 1.0),
        "traffic_uplift_x_drac_proxy": float(traffic_uplift_score) * drac,
        "proposed_rerouted_user_count": expected_acceptance_mass,
        "proposed_reroute_fraction": proposed_fraction,
        "proposed_rerouted_flow_vph": proposed_flow,
        "destination_edge_load_change": normalized_flow_delta,
        "maximum_edge_flow_delta": proposed_flow * max(0.25, 1.0 - base["route_overlap"]),
        "conflict_zone_exposure_delta": normalized_flow_delta * conflict_count,
        "lane_change_demand_delta": proposed_fraction
        * (base["lane_change_density"] + base["merge_interaction_density"]),
        "bottleneck_load_delta": normalized_flow_delta * base["bottleneck_centrality"],
        "route_overlap_delta": proposed_fraction * base["route_overlap"],
        "capacity_slack_delta": -normalized_flow_delta * vc,
        "expected_acceptance_mass": expected_acceptance_mass,
        "action_origin_merge": merge,
        "action_origin_signalized": signalized,
        "action_origin_two_route": two_route,
        "action_origin_asymmetric": asymmetric,
        "action_origin_real_like": real_like,
        "action_destination_primary_alternative": float(base["route_length_ratio"] <= 1.15),
        "action_destination_secondary_alternative": float(base["route_length_ratio"] > 1.15),
    }
    validate_action_features(features)
    return {name: float(features[name]) for name in ACTION_AWARE_FEATURE_SCHEMA}


def validate_action_features(features: Mapping[str, object]) -> None:
    if set(features) != set(ACTION_AWARE_FEATURE_SCHEMA):
        raise ValueError("v8 action-aware feature schema mismatch")
    for name, value in features.items():
        if any(token in name.lower() for token in FORBIDDEN_INPUT_TOKENS):
            raise ValueError(f"v8 forbidden post-decision feature: {name}")
        if not math.isfinite(float(value)):
            raise ValueError(f"v8 non-finite feature: {name}")


def feature_matrix(
    rows: Sequence[Mapping[str, object]],
    feature_names: Sequence[str] = ACTION_AWARE_FEATURE_SCHEMA,
) -> np.ndarray:
    return np.asarray(
        [[float(row["v8_features"][name]) for name in feature_names] for row in rows],
        dtype=float,
    )

