from __future__ import annotations

from typing import Mapping, Sequence

from concordia.alignment import compute_alignment_frontier
from concordia.feasibility.alignment_potential import compute_alignment_potential
from concordia.feasibility.features import extract_feasibility_features
from concordia.feasibility.labels import classify_alignment_case
from concordia.models import Route, User
from concordia.populations import generate_population
from concordia.preferences import UtilityModel, preference_slack
from concordia.scenarios import (
    merge_bottleneck,
    ring_road,
    signalized_intersection,
    two_route,
)


SCENARIOS = {
    "two_route": two_route,
    "ring": ring_road,
    "merge": merge_bottleneck,
    "signalized": signalized_intersection,
}


def _routes(network, od) -> dict[str, Route]:
    return {
        route.route_id: route
        for route in network.multiobjective_candidate_routes(
            *od, k_per_objective=4, max_overlap=1.0, pareto_filter=False
        )
    }


def _candidate_sets(
    routes: Mapping[str, Route], users: Sequence[User], penetration: float
) -> dict[str, tuple[str, ...]]:
    utility = UtilityModel()
    selected = round(len(users) * penetration)
    candidates = {}
    for index, user in enumerate(users):
        route_ids = tuple(routes)
        if index < selected:
            candidates[user.user_id] = route_ids
            continue
        utilities = utility.utilities(user.preferences, routes.values())
        private = min(sorted(route_ids), key=preference_slack(utilities).__getitem__)
        candidates[user.user_id] = (private,)
    return candidates


def build_alignment_case(
    *,
    scenario: str,
    seed: int,
    demand_scale: float,
    heterogeneity: str,
    navigation_penetration: float,
    user_count: int,
    regret_limit: float,
    epsilon_grid: Sequence[float],
    minimum_relative_ttt_gain: float,
    safety_delta: float,
    source_split: str,
    precomputed: Mapping[str, object] | None = None,
) -> dict:
    network, od, base_demand = SCENARIOS[scenario]()
    routes = _routes(network, od)
    users = generate_population(
        user_count, *od, heterogeneity, regret_limit, 5.0, int(seed)
    )
    candidates = _candidate_sets(routes, users, navigation_penetration)
    vehicle_flow = base_demand * demand_scale / user_count
    if precomputed is None:
        result = compute_alignment_frontier(
            network,
            routes,
            users,
            candidates,
            vehicle_flow,
            epsilon_grid,
            maximum_combinations=1_000_000,
        )
        points = [point.__dict__ for point in result.points]
        private_ttt = result.private_best_ttt
        eta_ttt = result.eta_only_ttt
        system_optimum = result.unconstrained_system_optimum_ttt
    else:
        points = list(precomputed["frontier"])
        private_ttt = float(precomputed["private_best_ttt"])
        eta_ttt = float(precomputed["eta_only_ttt"])
        system_optimum = float(precomputed["unconstrained_system_optimum_ttt"])
    by_epsilon = {round(float(point["epsilon"]), 8): point for point in points}
    adaptive = by_epsilon[round(regret_limit, 8)]
    baseline = by_epsilon[min(by_epsilon)]
    demand = base_demand * demand_scale
    alignment = compute_alignment_potential(
        network, routes, users, candidates, vehicle_flow, demand
    )
    safety_difference = float(adaptive["safety_risk"]) - float(baseline["safety_risk"])
    label = classify_alignment_case(
        baseline_ttt=eta_ttt,
        adaptive_ttt=float(adaptive["minimum_feasible_ttt"]),
        maximum_regret=float(adaptive["max_regret"]),
        regret_limit=regret_limit,
        baseline_risk=float(baseline["safety_risk"]),
        adaptive_risk=float(adaptive["safety_risk"]),
        safety_delta=safety_delta,
        legal=True,
        meaningful_intervention=int(adaptive["beneficial_diversion_count"]) > 0,
        minimum_relative_ttt_gain=minimum_relative_ttt_gain,
    )
    features = extract_feasibility_features(
        network=network,
        routes=routes,
        users=users,
        demand=demand,
        price_of_anarchy=private_ttt / max(system_optimum, 1e-12),
        alignment=alignment,
        acceptance_probability=float(adaptive["acceptance_rate"]),
        safety_margin=safety_delta - safety_difference,
        navigation_penetration=navigation_penetration,
    )
    case_id = (
        f"{source_split}-{scenario}-s{seed}-d{demand_scale:.3f}-"
        f"{heterogeneity}-p{navigation_penetration:.2f}"
    )
    return {
        "case_id": case_id,
        "scenario": scenario,
        "seed": int(seed),
        "features": features,
        "baseline_metrics": {
            "eta_only_ttt": eta_ttt,
            "private_best_ttt": private_ttt,
            "safety_risk": float(baseline["safety_risk"]),
            "legal": True,
        },
        "adaptive_counterfactual": {
            "ttt": float(adaptive["minimum_feasible_ttt"]),
            "relative_ttt_gain": (
                eta_ttt - float(adaptive["minimum_feasible_ttt"])
            )
            / max(eta_ttt, 1e-12),
            "maximum_regret": float(adaptive["max_regret"]),
            "safety_risk": float(adaptive["safety_risk"]),
            "safety_difference": safety_difference,
            "acceptance_probability": float(adaptive["acceptance_rate"]),
            "beneficial_diversion_count": int(adaptive["beneficial_diversion_count"]),
            "legal": True,
            "recommendation_integrity": "accepted-only execution",
        },
        "label": label.value,
        "source_split": source_split,
        "condition": {
            "demand_scale": float(demand_scale),
            "heterogeneity": heterogeneity,
            "navigation_penetration": float(navigation_penetration),
        },
    }
