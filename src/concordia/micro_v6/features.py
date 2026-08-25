from __future__ import annotations

from typing import Mapping

from concordia.errors import ValidationError


MICRO_V6_FEATURE_GROUPS = {
    "traffic_static": (
        "density_mean",
        "flow_mean",
        "occupancy_mean",
        "mean_speed",
        "speed_variance",
        "acceleration_variance",
        "queue_length",
        "halting_count",
        "lane_occupancy",
        "headway_mean",
        "headway_variance",
        "demand_vehicles_per_hour",
    ),
    "traffic_temporal": (
        "density_slope_30s",
        "speed_slope_30s",
        "flow_instability",
        "queue_growth_rate",
        "short_horizon_speed_oscillation",
        "occupancy_variance_30s",
    ),
    "topology": (
        "route_overlap",
        "alternative_capacity_ratio",
        "volume_capacity_ratio",
        "bottleneck_centrality",
        "route_length_ratio",
        "topology_merge",
        "topology_signalized",
        "topology_two_route",
        "topology_real_like",
        "topology_ring",
        "perturbation_strength",
    ),
    "preference": (
        "predicted_acceptance",
        "preference_slack_mean",
        "preference_slack_std",
        "preference_variance",
        "heterogeneity_low",
        "heterogeneity_medium",
        "heterogeneity_high",
        "heterogeneity_bimodal",
        "heterogeneity_long_tail",
        "acceptance_multiplier",
    ),
    "penetration": ("navigation_penetration",),
    "safety": (
        "minimum_headway",
        "closing_speed_p90",
        "drac_proxy_p95",
        "lane_change_density",
        "merge_interaction_density",
        "speed_differential",
        "hard_braking_recent_rate",
    ),
    "analytical": (
        "analytical_success_probability",
        "analytical_predicted_ttt_gain",
    ),
}

MICRO_V6_FEATURE_SCHEMA = tuple(
    name for group in MICRO_V6_FEATURE_GROUPS.values() for name in group
)

_FORBIDDEN_TOKENS = (
    "future",
    "realized",
    "outcome",
    "safe_micro_success",
    "adaptive_ttt",
    "safety_violation",
    "post_decision",
)


def validate_predecision_features(record: Mapping[str, object]) -> None:
    decision_time = float(record["decision_time"])
    observation_end = float(record["feature_observation_end_time"])
    if observation_end > decision_time + 1e-9:
        raise ValidationError("v6 feature window extends past the intervention decision")
    features = record["features_pre_decision"]
    if not isinstance(features, Mapping):
        raise ValidationError("v6 pre-decision features must be a mapping")
    if len(features) != len(MICRO_V6_FEATURE_SCHEMA) or set(features) != set(
        MICRO_V6_FEATURE_SCHEMA
    ):
        raise ValidationError("v6 micro feature schema mismatch")
    for name, value in features.items():
        lowered = name.lower()
        if any(token in lowered for token in _FORBIDDEN_TOKENS):
            raise ValidationError(f"future/outcome leakage in v6 feature: {name}")
        numeric = float(value)
        if numeric != numeric or numeric in (float("inf"), float("-inf")):
            raise ValidationError(f"non-finite v6 feature: {name}")
