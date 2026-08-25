from __future__ import annotations

import itertools
import math
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from concordia.adaptive.controller import ClosedLoopController
from concordia.behavior import AcceptanceModel
from concordia.evaluation import paired_comparison, summarize_samples
from concordia.models import AssignmentResult, Route, User
from concordia.optimization import (
    AdaptiveOptimizer,
    MIPAssignmentSolver,
    ObjectiveWeights,
    RecedingHorizonOptimizer,
)
from concordia.optimization.objective import route_concentration, upper_cvar
from concordia.populations import generate_population
from concordia.preferences import UtilityModel, preference_slack
from concordia.scenarios import merge_bottleneck, ring_road, signalized_intersection, two_route
from concordia.simulation import AnalyticalSimulationAdapter, analytical_edge_id


POLICY_LABELS = {
    "B0": "static_free_flow_fastest",
    "B1": "dynamic_eta_only_best_response",
    "B2": "preference_only_private_best",
    "B3": "small_instance_system_optimal_oracle",
    "B4": "greedy_vde",
    "B5": "linearized_constrained_mip",
    "B6": "receding_horizon_acceptance_aware_closed_loop",
}

SCENARIOS = {
    "two_route": two_route,
    "merge": merge_bottleneck,
    "signalized": signalized_intersection,
    "ring": ring_road,
}


def _flows(optimizer: AdaptiveOptimizer, assignments: Mapping[str, str]):
    return optimizer._flows(assignments)


def _regrets(
    users: Sequence[User],
    candidates: Mapping[str, Sequence[str]],
    routes: Mapping[str, Route],
    assignments: Mapping[str, str],
) -> Dict[str, float]:
    utility_model = UtilityModel()
    result = {}
    for user in users:
        utilities = utility_model.utilities(
            user.preferences, (routes[route_id] for route_id in candidates[user.user_id])
        )
        result[user.user_id] = preference_slack(utilities)[assignments[user.user_id]]
    return result


def _dynamic_eta_assignment(
    optimizer: AdaptiveOptimizer,
    users: Sequence[User],
    candidates: Mapping[str, Sequence[str]],
) -> AssignmentResult:
    fastest = min(optimizer.routes, key=lambda route_id: optimizer.routes[route_id].features.time)
    assignments = {user.user_id: fastest for user in users}
    for _ in range(100):
        changed = False
        for user in users:
            best_route = assignments[user.user_id]
            best_key = None
            for route_id in candidates[user.user_id]:
                trial = {**assignments, user.user_id: route_id}
                flows = _flows(optimizer, trial)
                eta = optimizer.network.path_features(optimizer.routes[route_id].nodes, flows).time
                key = (eta, route_id)
                if best_key is None or key < best_key:
                    best_key = key
                    best_route = route_id
            if best_route != assignments[user.user_id]:
                assignments[user.user_id] = best_route
                changed = True
        if not changed:
            break
    return optimizer.evaluate(assignments, _regrets(users, candidates, optimizer.routes, assignments))


def _system_optimal_oracle(
    optimizer: AdaptiveOptimizer,
    users: Sequence[User],
    candidates: Mapping[str, Sequence[str]],
    maximum_combinations: int = 100_000,
) -> AssignmentResult:
    option_lists = [tuple(candidates[user.user_id]) for user in users]
    combinations = math.prod(len(options) for options in option_lists)
    if combinations > maximum_combinations:
        raise ValueError("system-optimal oracle exceeds its declared small-instance limit")
    best = None
    for selected in itertools.product(*option_lists):
        assignments = {user.user_id: route for user, route in zip(users, selected)}
        evaluated = optimizer.evaluate(assignments)
        key = (evaluated.total_travel_time, tuple(sorted(assignments.items())))
        if best is None or key < best[0]:
            best = (key, assignments)
    assert best is not None
    assignments = best[1]
    return optimizer.evaluate(assignments, _regrets(users, candidates, optimizer.routes, assignments))


def _mip_assignment(
    optimizer: AdaptiveOptimizer,
    users: Sequence[User],
    candidates: Mapping[str, Sequence[str]],
    baseline: AssignmentResult,
) -> tuple[AssignmentResult, str]:
    utility_model = UtilityModel()
    utilities = {
        user.user_id: utility_model.utilities(
            user.preferences, (optimizer.routes[route_id] for route_id in candidates[user.user_id])
        )
        for user in users
    }
    current_flows = _flows(optimizer, baseline.assignments)
    costs = {}
    acceptance = {}
    for user in users:
        for route_id in candidates[user.user_id]:
            route = optimizer.routes[route_id]
            costs[(user.user_id, route_id)] = sum(
                optimizer.network.edge_data(edge).travel_time(current_flows[edge])
                for edge in route.edges
            )
            acceptance[(user.user_id, route_id)] = 1.0
    result = MIPAssignmentSolver(optimizer.network, optimizer.routes).solve(
        users,
        candidates,
        utilities,
        costs,
        acceptance,
        baseline.assignments,
        safety_delta=0.0,
    )
    evaluated = optimizer.evaluate(result.assignments, result.regrets)
    return evaluated, result.solver


def _closed_loop_assignment(
    optimizer: AdaptiveOptimizer,
    users: Sequence[User],
    candidates: Mapping[str, Sequence[str]],
    initial: AssignmentResult,
    seed: int,
) -> tuple[AssignmentResult, Dict[str, float]]:
    simulator = AnalyticalSimulationAdapter(
        optimizer.network,
        optimizer.routes,
        initial.assignments,
        optimizer.vehicle_flow,
    )
    acceptance_model = AcceptanceModel()
    mpc = RecedingHorizonOptimizer(
        optimizer.network,
        optimizer.routes,
        optimizer.vehicle_flow,
        horizon_steps=3,
        minimum_acceptance_probability=0.0,
        safety_delta=0.0,
        weights=ObjectiveWeights(ghost_risk=0.01, safety_risk=1.0, concentration=1.0),
        acceptance_model=acceptance_model,
    )
    controller = ClosedLoopController(
        optimizer.network,
        optimizer.routes,
        users,
        candidates,
        initial.assignments,
        simulator,
        simulator.edge_id_map,
        mpc,
        acceptance_model,
        seed,
        {
            route_id: tuple(analytical_edge_id(edge) for edge in route.edges)
            for route_id, route in optimizer.routes.items()
        },
    )
    result = controller.run(steps=3)
    evaluated = optimizer.evaluate(
        result.final_assignments,
        _regrets(users, candidates, optimizer.routes, result.final_assignments),
    )
    offers = sum(len(step.decisions) for step in result.steps)
    switches = sum(step.accepted_count for step in result.steps)
    last_departed_route: Dict[str, str] = {}
    reversals = 0
    for step in result.steps:
        for decision in step.decisions:
            if not decision.accepted:
                continue
            user_id = decision.offer.user_id
            if decision.offer.candidate_route_id == last_departed_route.get(user_id):
                reversals += 1
            last_departed_route[user_id] = decision.offer.current_route_id
    return evaluated, {
        "acceptance_rate": result.acceptance_rate,
        "offer_count": float(offers),
        "accepted_switch_count": float(switches),
        "route_reversal_count": float(reversals),
        "mean_solve_time_seconds": float(
            np.mean([step.plan.solve_time_seconds for step in result.steps])
        ),
    }


def _policy_metrics(
    result: AssignmentResult,
    routes: Mapping[str, Route],
    elapsed_seconds: float,
    peak_memory_bytes: int,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    hhi, entropy = route_concentration(result.assignments)
    risks = [float(value) for value in result.regrets.values()]
    route_risks = [routes[route_id].features.risk for route_id in result.assignments.values()]
    return {
        "total_travel_time_vehicle_minutes_per_hour": result.total_travel_time,
        "route_surrogate_risk_total": result.total_safety_risk,
        "route_surrogate_risk_cvar95": upper_cvar(route_risks),
        "route_entropy": entropy,
        "route_hhi": hhi,
        "mean_regret": float(np.mean(risks)),
        "p95_regret": float(np.percentile(risks, 95)),
        "max_regret": float(np.max(risks)),
        "preference_slack_consumed": float(np.sum(risks)),
        "latency_seconds": elapsed_seconds,
        "peak_python_memory_bytes": peak_memory_bytes,
        "assignments": dict(result.assignments),
        **dict(extra or {}),
    }


def evaluate_policy_suite(
    scenario: str,
    demand_scale: float,
    heterogeneity: str,
    epsilon: float,
    seed: int,
    user_count: int = 8,
) -> Dict[str, Any]:
    network, od, base_demand = SCENARIOS[scenario]()
    routes_list = network.multiobjective_candidate_routes(
        od[0], od[1], k_per_objective=4, max_overlap=1.0
    )
    routes = {route.route_id: route for route in routes_list}
    users = generate_population(
        user_count, od[0], od[1], heterogeneity, epsilon, 5.0, seed
    )
    candidates = {user.user_id: tuple(routes) for user in users}
    optimizer = AdaptiveOptimizer(
        network,
        routes,
        objective_weights=ObjectiveWeights(ghost_risk=0.01, safety_risk=1.0, concentration=1.0),
        vehicle_flow=(base_demand * demand_scale) / user_count,
    )

    policy_builders: Dict[str, Any] = {}
    fastest = min(routes, key=lambda route_id: routes[route_id].features.time)
    static_assignments = {user.user_id: fastest for user in users}
    policy_builders["B0"] = lambda: optimizer.evaluate(
        static_assignments, _regrets(users, candidates, routes, static_assignments)
    )
    policy_builders["B1"] = lambda: _dynamic_eta_assignment(optimizer, users, candidates)
    policy_builders["B2"] = lambda: optimizer.private_best(users, candidates)
    policy_builders["B3"] = lambda: _system_optimal_oracle(optimizer, users, candidates)
    policy_builders["B4"] = lambda: optimizer.greedy_vde(users, candidates, safety_delta=0.0)

    outputs: Dict[str, Any] = {}
    raw_results: Dict[str, AssignmentResult] = {}
    for policy in ("B0", "B1", "B2", "B3", "B4"):
        tracemalloc.start()
        started = time.perf_counter()
        raw_results[policy] = policy_builders[policy]()
        elapsed = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        outputs[policy] = _policy_metrics(raw_results[policy], routes, elapsed, peak)

    tracemalloc.start()
    started = time.perf_counter()
    mip_result, solver = _mip_assignment(
        optimizer, users, candidates, raw_results["B2"]
    )
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    outputs["B5"] = _policy_metrics(mip_result, routes, elapsed, peak, {"solver": solver})

    tracemalloc.start()
    started = time.perf_counter()
    closed_loop, closed_loop_extra = _closed_loop_assignment(
        optimizer, users, candidates, raw_results["B2"], seed
    )
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    outputs["B6"] = _policy_metrics(closed_loop, routes, elapsed, peak, closed_loop_extra)

    beneficial_opportunities = 0
    baseline = raw_results["B2"]
    baseline_ttt = baseline.total_travel_time
    for user in users:
        utilities = UtilityModel().utilities(
            user.preferences, (routes[route_id] for route_id in candidates[user.user_id])
        )
        slacks = preference_slack(utilities)
        for route_id in candidates[user.user_id]:
            if route_id == baseline.assignments[user.user_id] or slacks[route_id] > epsilon + 1e-10:
                continue
            trial = optimizer.evaluate({**baseline.assignments, user.user_id: route_id})
            if trial.total_travel_time < baseline_ttt - 1e-9:
                beneficial_opportunities += 1
                break
    preference_matrix = np.asarray(
        [list(user.preferences.normalized().as_dict().values()) for user in users], dtype=float
    )
    return {
        "scenario": scenario,
        "seed": seed,
        "demand_scale": demand_scale,
        "base_demand_vehicles_per_hour": base_demand,
        "heterogeneity": heterogeneity,
        "utility_epsilon": epsilon,
        "candidate_route_count": len(routes),
        "beneficial_diversion_opportunities": beneficial_opportunities,
        "preference_mean_time_weight": float(preference_matrix[:, 0].mean()),
        "preference_total_variance": float(preference_matrix.var(axis=0).sum()),
        "preference_time_weight_p90": float(np.percentile(preference_matrix[:, 0], 90)),
        "policies": outputs,
    }


def evaluate_screening_case(
    scenario: str,
    demand_scale: float,
    heterogeneity: str,
    epsilon: float,
    seed: int,
    user_count: int = 8,
) -> Dict[str, Any]:
    """Lightweight B2/B4 case used only to select focused factors."""
    network, od, base_demand = SCENARIOS[scenario]()
    routes = {
        route.route_id: route
        for route in network.multiobjective_candidate_routes(
            od[0], od[1], k_per_objective=4, max_overlap=1.0
        )
    }
    users = generate_population(
        user_count, od[0], od[1], heterogeneity, epsilon, 5.0, seed
    )
    candidates = {user.user_id: tuple(routes) for user in users}
    optimizer = AdaptiveOptimizer(
        network,
        routes,
        objective_weights=ObjectiveWeights(ghost_risk=0.01, safety_risk=1.0, concentration=1.0),
        vehicle_flow=(base_demand * demand_scale) / user_count,
    )
    private = optimizer.private_best(users, candidates)
    greedy = optimizer.greedy_vde(users, candidates, safety_delta=0.0)
    opportunities = 0
    for user in users:
        utilities = UtilityModel().utilities(
            user.preferences, (routes[route_id] for route_id in candidates[user.user_id])
        )
        slacks = preference_slack(utilities)
        for route_id in candidates[user.user_id]:
            if route_id == private.assignments[user.user_id] or slacks[route_id] > epsilon + 1e-10:
                continue
            trial = optimizer.evaluate({**private.assignments, user.user_id: route_id})
            if trial.total_travel_time < private.total_travel_time - 1e-9:
                opportunities += 1
                break
    matrix = np.asarray(
        [list(user.preferences.normalized().as_dict().values()) for user in users], dtype=float
    )
    return {
        "scenario": scenario,
        "seed": seed,
        "demand_scale": demand_scale,
        "heterogeneity": heterogeneity,
        "utility_epsilon": epsilon,
        "candidate_route_count": len(routes),
        "beneficial_diversion_opportunities": opportunities,
        "preference_mean_time_weight": float(matrix[:, 0].mean()),
        "preference_total_variance": float(matrix.var(axis=0).sum()),
        "preference_time_weight_p90": float(np.percentile(matrix[:, 0], 90)),
        "b2_ttt": private.total_travel_time,
        "b4_ttt": greedy.total_travel_time,
    }


def _r_squared(features: np.ndarray, target: np.ndarray) -> float:
    design = np.column_stack([np.ones(len(features)), features])
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    prediction = design @ coefficients
    total = float(np.sum((target - target.mean()) ** 2))
    return 0.0 if math.isclose(total, 0.0) else 1.0 - float(np.sum((target - prediction) ** 2)) / total


def run_research_matrix(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Run declared screening then a focused paired analytical matrix."""
    seeds = [int(seed) for seed in config["seeds"]]
    screening_rows = []
    for scenario, scale, heterogeneity, epsilon, seed in itertools.product(
        config["screening"]["scenarios"],
        config["screening"]["demand_scale"],
        config["screening"]["heterogeneity"],
        config["screening"]["utility_epsilon"],
        seeds,
    ):
        screening_rows.append(
            evaluate_screening_case(
                scenario,
                float(scale),
                str(heterogeneity),
                float(epsilon),
                seed,
                int(config["screening"].get("user_count", 8)),
            )
        )

    focused_rows = []
    for scenario, scale, seed in itertools.product(
        config["focused"]["scenarios"], config["focused"]["demand_scale"], seeds
    ):
        focused_rows.append(
            evaluate_policy_suite(
                scenario,
                float(scale),
                str(config["focused"]["heterogeneity"]),
                float(config["focused"]["utility_epsilon"]),
                seed,
                int(config["focused"].get("user_count", 6)),
            )
        )

    b1 = [row["policies"]["B1"]["total_travel_time_vehicle_minutes_per_hour"] for row in focused_rows]
    b6 = [row["policies"]["B6"]["total_travel_time_vehicle_minutes_per_hour"] for row in focused_rows]
    b4 = [row["policies"]["B4"]["total_travel_time_vehicle_minutes_per_hour"] for row in focused_rows]
    h1_low = [row["beneficial_diversion_opportunities"] for row in screening_rows if row["heterogeneity"] in {"none", "low"}]
    h1_high = [row["beneficial_diversion_opportunities"] for row in screening_rows if row["heterogeneity"] in {"high", "long_tail"}]
    explanatory_target = np.asarray(
        [row["b2_ttt"] - row["b4_ttt"] for row in screening_rows], dtype=float
    )
    mean_features = np.asarray(
        [[row["preference_mean_time_weight"]] for row in screening_rows], dtype=float
    )
    tail_features = np.asarray(
        [
            [row["preference_total_variance"], row["preference_time_weight_p90"]]
            for row in screening_rows
        ],
        dtype=float,
    )
    max_regret = max(row["policies"]["B6"]["max_regret"] for row in focused_rows)
    h2 = paired_comparison(b1, b6, seed=901)
    h2["all_b6_regret_constraints_met"] = bool(
        max_regret <= float(config["focused"]["utility_epsilon"]) + 1e-10
    )
    h2["maximum_b6_regret"] = max_regret
    return {
        "study_type": "synthetic_analytical_screening_and_focused_paired_matrix",
        "claim_boundary": (
            "B1-B6 are analytical BPR policy comparisons. Route risk is a surrogate; "
            "these rows are not microscopic traffic or crash evidence."
        ),
        "policy_definitions": POLICY_LABELS,
        "screening_row_count": len(screening_rows),
        "focused_row_count": len(focused_rows),
        "screening": screening_rows,
        "focused": focused_rows,
        "hypotheses": {
            "H1": {
                "primary_metric": "beneficial_diversion_opportunities_per_population",
                "low_or_none": summarize_samples(h1_low, seed=911),
                "high_or_long_tail": summarize_samples(h1_high, seed=912),
                "status": "supported" if np.mean(h1_high) > np.mean(h1_low) else "not_supported",
            },
            "H2": {
                "primary_metric": "B1_minus_B6_TTT_under_regret_constraint",
                **h2,
                "status": (
                    "supported"
                    if h2["bootstrap_ci95_low"] is not None
                    and h2["bootstrap_ci95_low"] > 0
                    and h2["all_b6_regret_constraints_met"]
                    else "not_supported"
                ),
            },
            "H5": {
                "primary_metric": "out_of_sample_explanatory_analysis_not_available",
                "mean_weight_in_sample_r_squared": _r_squared(mean_features, explanatory_target),
                "variance_tail_in_sample_r_squared": _r_squared(tail_features, explanatory_target),
                "status": "exploratory_only",
                "limitation": "in-sample screening association; not causal or out-of-sample evidence",
            },
            "H6": {
                "primary_metric": "B4_vs_B6_TTT_variability_and_constraint_violations",
                "b4_coefficient_of_variation": float(np.std(b4, ddof=1) / np.mean(b4)),
                "b6_coefficient_of_variation": float(np.std(b6, ddof=1) / np.mean(b6)),
                "status": "not_supported",
                "limitation": (
                    "Analytical three-step feedback did not pre-register a validated stability "
                    "threshold; result is descriptive."
                ),
            },
        },
    }


def write_research_summary(summary: Mapping[str, Any], output: str) -> Path:
    import json

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination
