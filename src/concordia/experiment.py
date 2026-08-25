from __future__ import annotations

import random
from typing import Any, Dict, Mapping

from concordia.evaluation import paired_effect_size, summarize_samples
from concordia.explainability import explain_recommendation
from concordia.models import Route
from concordia.optimization import AdaptiveOptimizer, ObjectiveWeights
from concordia.populations import generate_population
from concordia.scenarios import braess, merge_bottleneck, ring_road, signalized_intersection, two_route
from concordia.traffic import TrafficAssignment


def _scenario(name: str):
    return {
        "two_route": two_route,
        "braess": braess,
        "ring": ring_road,
        "merge": merge_bottleneck,
        "signalized": signalized_intersection,
    }[name]()


def run_experiment(config: Mapping[str, Any]) -> Dict[str, Any]:
    network, od, base_demand = _scenario(str(config["scenario"]))
    demand = base_demand * float(config["demand_scale"])
    user_count = int(config.get("user_count", 8))
    route_list = network.candidate_routes(od[0], od[1], k=int(config.get("candidate_count", 4)), max_overlap=1.0)
    routes: Dict[str, Route] = {route.route_id: route for route in route_list}
    all_route_ids = tuple(routes)
    candidates = {f"u{index:04d}": all_route_ids for index in range(user_count)}
    static_ttt = []
    baseline_ttt = []
    proposed_ttt = []
    proposed_regret = []
    entropy = []
    safety = []
    decision_log = []
    for seed in config["seeds"]:
        users = generate_population(
            count=user_count,
            origin=od[0],
            destination=od[1],
            heterogeneity=str(config["population"]),
            epsilon=float(config["utility_epsilon"]),
            rationality=float(config.get("rationality", 5.0)),
            seed=int(seed),
        )
        optimizer = AdaptiveOptimizer(
            network,
            routes,
            objective_weights=ObjectiveWeights(
                ghost_risk=float(config.get("ghost_weight", 0.0)),
                safety_risk=float(config.get("safety_weight", 0.0)),
                concentration=float(config.get("concentration_weight", 0.0)),
            ),
            vehicle_flow=demand / user_count,
        )
        baseline = optimizer.private_best(users, candidates)
        static = optimizer.evaluate(
            {user.user_id: all_route_ids[0] for user in users},
            {user.user_id: 0.0 for user in users},
        )
        penetration = float(config["navigation_penetration"])
        navigated_count = int(round(user_count * penetration))
        navigated_ids = [user.user_id for user in users]
        random.Random(int(seed)).shuffle(navigated_ids)
        navigated = set(navigated_ids[:navigated_count])
        policy_candidates = {
            user.user_id: (
                candidates[user.user_id]
                if user.user_id in navigated
                else (baseline.assignments[user.user_id],)
            )
            for user in users
        }
        if config["policy"] == "exact":
            proposed = optimizer.exact(
                users,
                policy_candidates,
                safety_delta=float(config.get("safety_delta", 0.0)),
            )
        else:
            proposed = optimizer.greedy_vde(
                users,
                policy_candidates,
                safety_delta=float(config.get("safety_delta", 0.0)),
            )
        static_ttt.append(static.total_travel_time)
        baseline_ttt.append(baseline.total_travel_time)
        proposed_ttt.append(proposed.total_travel_time)
        proposed_regret.extend(proposed.regrets.values())
        entropy.append(proposed.route_entropy)
        safety.append(proposed.total_safety_risk)
        for user in users:
            route_candidates = [routes[route_id] for route_id in candidates[user.user_id]]
            utilities = optimizer.utility_model.utilities(user.preferences, route_candidates)
            best_utility = max(utilities.values())
            slacks = {route_id: best_utility - value for route_id, value in utilities.items()}
            selected_id = proposed.assignments[user.user_id]
            reference_id = baseline.assignments[user.user_id]
            counterfactual_assignments = dict(proposed.assignments)
            counterfactual_assignments[user.user_id] = reference_id
            counterfactual = optimizer.evaluate(counterfactual_assignments)
            explanation = explain_recommendation(
                selected=routes[selected_id],
                reference=routes[reference_id],
                selected_utility=utilities[selected_id],
                reference_utility=utilities[reference_id],
                estimated_network_benefit=(
                    counterfactual.total_travel_time - proposed.total_travel_time
                ),
            )
            decision_log.append(
                {
                    "seed": int(seed),
                    "user_id": user.user_id,
                    "candidate_routes": list(candidates[user.user_id]),
                    "estimated_utilities": utilities,
                    "preference_slack": slacks,
                    "selected_route": selected_id,
                    "policy": config["policy"],
                    "navigation_eligible": user.user_id in navigated,
                    "selected_route_safety_risk": routes[selected_id].features.risk,
                    "explanation": explanation.as_dict(),
                }
            )
    assignment = TrafficAssignment(network)
    ue = assignment.user_equilibrium({od: demand})
    so = assignment.system_optimum({od: demand})
    return {
        "scenario": config["scenario"],
        "synthetic": True,
        "demand": demand,
        "ue": {
            "total_travel_time": ue.total_travel_time,
            "relative_gap": ue.relative_gap,
            "converged": ue.converged,
        },
        "so": {
            "total_travel_time": so.total_travel_time,
            "relative_gap": so.relative_gap,
            "converged": so.converged,
        },
        "price_of_anarchy": ue.total_travel_time / so.total_travel_time,
        "b0_static_shortest_ttt": summarize_samples(static_ttt, seed=100),
        "b1_dynamic_eta_analytical_proxy": {
            "total_travel_time": ue.total_travel_time,
            "limitation": "static Wardrop proxy; time-varying B1 requires SUMO",
        },
        "private_best_ttt": summarize_samples(baseline_ttt, seed=101),
        "proposed_ttt": summarize_samples(proposed_ttt, seed=102),
        "paired_effect_size": paired_effect_size(baseline_ttt, proposed_ttt),
        "proposed_regret": summarize_samples(proposed_regret, seed=103),
        "route_entropy": summarize_samples(entropy, seed=104),
        "surrogate_route_risk": summarize_samples(safety, seed=105),
        "decision_log": decision_log,
    }
