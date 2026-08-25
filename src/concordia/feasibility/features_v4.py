from __future__ import annotations

from typing import Mapping

import numpy as np

from concordia.feasibility.features import FEATURE_SCHEMA
from concordia.models import FEATURE_NAMES
from concordia.populations import generate_population
from concordia.preferences import UtilityModel, preference_slack
from concordia.scenarios import merge_bottleneck, ring_road, signalized_intersection, two_route


SCENARIOS = {
    "two_route": two_route,
    "ring": ring_road,
    "merge": merge_bottleneck,
    "signalized": signalized_intersection,
}

V4_ADDITIONAL_FEATURES = (
    "route_count",
    "bottleneck_load",
    "beneficial_alternative_count",
    "mean_preference_slack",
    "p90_preference_slack",
    "preference_long_tail_mass",
    "eta_dispersion",
    "reliability_dispersion",
    "safety_dispersion",
    "cost_dispersion",
    "aps_alternative_capacity_interaction",
    "route_overlap_demand_interaction",
    "poa_penetration_interaction",
    "safety_margin_demand_interaction",
)

V4_FEATURE_SCHEMA = FEATURE_SCHEMA + V4_ADDITIONAL_FEATURES


def _coefficient_of_variation(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    return float(array.std() / max(abs(float(array.mean())), 1e-9))


def expand_v4_features(case: Mapping[str, object]) -> dict[str, float]:
    """Expand a v3 case without changing the frozen v3 feature contract."""
    scenario = str(case["scenario"])
    seed = int(case["seed"])
    condition = case["condition"]
    heterogeneity = str(condition["heterogeneity"])
    network, od, _base_demand = SCENARIOS[scenario]()
    routes = network.multiobjective_candidate_routes(
        *od, k_per_objective=4, max_overlap=1.0, pareto_filter=False
    )
    users = generate_population(6, *od, heterogeneity, 0.08, 5.0, seed)
    utility_model = UtilityModel()
    slack_values = []
    for user in users:
        utilities = utility_model.utilities(user.preferences, routes)
        slacks = preference_slack(utilities)
        private = min(sorted(slacks), key=slacks.__getitem__)
        slack_values.extend(float(value) for route_id, value in slacks.items() if route_id != private)
    route_features = [route.features.as_dict() for route in routes]
    normalized_weights = np.asarray(
        [
            [getattr(user.preferences.normalized(), name) for name in FEATURE_NAMES]
            for user in users
        ],
        dtype=float,
    )
    median_weight = float(np.median(normalized_weights))
    long_tail_mass = float(np.mean(normalized_weights > 3.0 * max(median_weight, 1e-9)))
    base = {name: float(case["features"][name]) for name in FEATURE_SCHEMA}
    additions = {
        "route_count": float(len(routes)),
        "bottleneck_load": base["volume_capacity_ratio"]
        * (1.0 + base["bottleneck_centrality"]),
        "beneficial_alternative_count": base["alignment_opportunity_count"],
        "mean_preference_slack": float(np.mean(slack_values)) if slack_values else 0.0,
        "p90_preference_slack": float(np.percentile(slack_values, 90)) if slack_values else 0.0,
        "preference_long_tail_mass": long_tail_mass,
        "eta_dispersion": _coefficient_of_variation(
            [float(value["time"]) for value in route_features]
        ),
        "reliability_dispersion": _coefficient_of_variation(
            [float(value["variability"]) for value in route_features]
        ),
        "safety_dispersion": _coefficient_of_variation(
            [float(value["risk"]) for value in route_features]
        ),
        "cost_dispersion": _coefficient_of_variation(
            [float(value["cost"]) for value in route_features]
        ),
        "aps_alternative_capacity_interaction": base["alignment_potential_score"]
        * base["alternative_capacity_ratio"],
        "route_overlap_demand_interaction": base["route_overlap"] * base["demand"],
        "poa_penetration_interaction": base["price_of_anarchy"]
        * base["navigation_penetration"],
        "safety_margin_demand_interaction": base["safety_margin"] * base["demand"],
    }
    features = {**base, **additions}
    return {name: float(features[name]) for name in V4_FEATURE_SCHEMA}
