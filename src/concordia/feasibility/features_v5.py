from __future__ import annotations

from typing import Mapping

from concordia.feasibility.features_v4 import V4_FEATURE_SCHEMA, expand_v4_features


V5_ADDITIONAL_FEATURES = (
    "penetration_aps_interaction",
    "penetration_alternative_capacity_interaction",
    "penetration_route_overlap_interaction",
    "dss_penetration_interaction",
)

V5_FEATURE_SCHEMA = V4_FEATURE_SCHEMA + V5_ADDITIONAL_FEATURES


def expand_v5_features(
    case: Mapping[str, object], *, domain_shift_score: float = 0.0
) -> dict[str, float]:
    base = expand_v4_features(case)
    penetration = base["navigation_penetration"]
    additions = {
        "penetration_aps_interaction": penetration
        * base["alignment_potential_score"],
        "penetration_alternative_capacity_interaction": penetration
        * base["alternative_capacity_ratio"],
        "penetration_route_overlap_interaction": penetration
        * base["route_overlap"],
        "dss_penetration_interaction": float(domain_shift_score) * penetration,
    }
    features = {**base, **additions}
    return {name: float(features[name]) for name in V5_FEATURE_SCHEMA}
