from __future__ import annotations

import itertools
import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from concordia.behavior import AcceptanceModel
from concordia.errors import ValidationError
from concordia.models import EdgeKey, Route, User
from concordia.network import RoadNetwork
from concordia.preferences import UtilityModel, preference_slack


@dataclass(frozen=True)
class AlignmentFrontierPoint:
    epsilon: float
    minimum_feasible_ttt: float
    price_of_alignment: float
    mean_regret: float
    p95_regret: float
    max_regret: float
    acceptance_rate: float
    beneficial_diversion_count: int
    route_entropy: float
    safety_risk: float
    marginal_cost_reduction_per_epsilon: float
    alignment_region: str


@dataclass(frozen=True)
class AlignmentFrontierResult:
    points: Tuple[AlignmentFrontierPoint, ...]
    unconstrained_system_optimum_ttt: float
    private_best_ttt: float
    eta_only_ttt: float
    knee_epsilon: float
    knee_index: int
    monotonic: bool
    h1_robustness: Mapping[str, float]


def _flows(
    network: RoadNetwork,
    routes: Mapping[str, Route],
    assignment: Mapping[str, str],
    vehicle_flow: float,
) -> Dict[EdgeKey, float]:
    result = {edge: 0.0 for edge in network.edges}
    for route_id in assignment.values():
        for edge in routes[route_id].edges:
            result[edge] += vehicle_flow
    return result


def _ttt(network: RoadNetwork, flows: Mapping[EdgeKey, float]) -> float:
    return sum(
        flow * network.edge_data(edge).travel_time(flow) for edge, flow in flows.items()
    )


def _eta_only_assignment(
    network: RoadNetwork,
    routes: Mapping[str, Route],
    users: Sequence[User],
    candidates: Mapping[str, Sequence[str]],
    vehicle_flow: float,
) -> tuple[Mapping[str, str], float]:
    fastest = min(routes, key=lambda route_id: routes[route_id].features.time)
    assignment = {user.user_id: fastest for user in users}
    for _ in range(100):
        changed = False
        for user in users:
            choice = min(
                candidates[user.user_id],
                key=lambda route_id: (
                    network.path_features(
                        routes[route_id].nodes,
                        _flows(
                            network,
                            routes,
                            {**assignment, user.user_id: route_id},
                            vehicle_flow,
                        ),
                    ).time,
                    route_id,
                ),
            )
            if choice != assignment[user.user_id]:
                assignment[user.user_id] = choice
                changed = True
        if not changed:
            break
    flows = _flows(network, routes, assignment, vehicle_flow)
    return assignment, _ttt(network, flows)


def _entropy(assignment: Mapping[str, str]) -> float:
    counts = Counter(assignment.values())
    total = len(assignment)
    return -sum((count / total) * math.log(count / total) for count in counts.values())


def _knee(epsilon: np.ndarray, cost: np.ndarray) -> tuple[int, float]:
    if len(epsilon) < 3 or np.ptp(cost) <= 1e-12:
        return 0, float(epsilon[0])
    x = (epsilon - epsilon.min()) / max(float(np.ptp(epsilon)), 1e-12)
    y = (cost - cost.min()) / max(float(np.ptp(cost)), 1e-12)
    first = np.gradient(y, x)
    curvature = np.abs(np.gradient(first, x)) / np.maximum(
        (1.0 + first**2) ** 1.5, 1e-12
    )
    curvature[0] = curvature[-1] = -np.inf
    index = int(np.argmax(curvature))
    return index, float(epsilon[index])


def compute_alignment_frontier(
    network: RoadNetwork,
    routes: Mapping[str, Route],
    users: Sequence[User],
    candidates: Mapping[str, Sequence[str]],
    vehicle_flow: float,
    epsilon_grid: Sequence[float],
    maximum_combinations: int = 1_000_000,
) -> AlignmentFrontierResult:
    if not users or vehicle_flow <= 0 or not epsilon_grid:
        raise ValidationError("alignment frontier inputs are invalid")
    epsilon = np.asarray(sorted(set(float(item) for item in epsilon_grid)), dtype=float)
    if np.any(epsilon < 0):
        raise ValidationError("alignment epsilon cannot be negative")
    utility_model = UtilityModel()
    acceptance_model = AcceptanceModel()
    slacks = {}
    utilities = {}
    option_lists = []
    combinations = 1
    for user in users:
        options = tuple(candidates[user.user_id])
        option_lists.append(options)
        combinations *= len(options)
        if combinations > maximum_combinations:
            raise ValidationError("alignment oracle exceeds its declared combination limit")
        utilities[user.user_id] = utility_model.utilities(
            user.preferences, (routes[route_id] for route_id in options)
        )
        slacks[user.user_id] = preference_slack(utilities[user.user_id])

    evaluated = []
    for selected in itertools.product(*option_lists):
        assignment = {user.user_id: route_id for user, route_id in zip(users, selected)}
        regrets = {
            user.user_id: slacks[user.user_id][assignment[user.user_id]] for user in users
        }
        flows = _flows(network, routes, assignment, vehicle_flow)
        evaluated.append(
            {
                "assignment": assignment,
                "regrets": regrets,
                "ttt": _ttt(network, flows),
                "safety": sum(routes[route_id].features.risk for route_id in assignment.values()),
            }
        )
    system_optimum = min(evaluated, key=lambda item: (item["ttt"], sorted(item["assignment"].items())))
    private_assignment = {
        user.user_id: min(
            sorted(candidates[user.user_id]), key=slacks[user.user_id].__getitem__
        )
        for user in users
    }
    private = next(item for item in evaluated if item["assignment"] == private_assignment)
    _, eta_ttt = _eta_only_assignment(
        network, routes, users, candidates, vehicle_flow
    )

    raw_points = []
    final_route_diversity = 0
    for bound in epsilon:
        feasible = [
            item
            for item in evaluated
            if max(item["regrets"].values(), default=0.0) <= bound + 1e-10
        ]
        if not feasible:
            raise ValidationError("private-best assignment must make every epsilon feasible")
        best = min(feasible, key=lambda item: (item["ttt"], sorted(item["assignment"].items())))
        regret_values = np.asarray(list(best["regrets"].values()), dtype=float)
        diversions = sum(
            best["assignment"][user.user_id] != private_assignment[user.user_id]
            for user in users
        )
        final_route_diversity = len(set(best["assignment"].values()))
        offered_probabilities = []
        for user in users:
            selected_id = best["assignment"][user.user_id]
            current_id = private_assignment[user.user_id]
            if selected_id == current_id:
                continue
            selected = routes[selected_id].features
            current = routes[current_id].features
            offered_probabilities.append(
                acceptance_model.probability(
                    best["regrets"][user.user_id],
                    utilities[user.user_id][selected_id] - utilities[user.user_id][current_id],
                    current.time - selected.time,
                    current.variability - selected.variability,
                    max(0.0, private["ttt"] - best["ttt"]),
                )
            )
        region = (
            "WIN"
            if diversions and best["ttt"] < eta_ttt - 1e-9
            else "TRADEOFF"
            if diversions
            else "INFEASIBLE"
        )
        raw_points.append(
            {
                "epsilon": float(bound),
                "minimum_feasible_ttt": float(best["ttt"]),
                "price_of_alignment": float(best["ttt"] / system_optimum["ttt"]),
                "mean_regret": float(regret_values.mean()),
                "p95_regret": float(np.percentile(regret_values, 95)),
                "max_regret": float(regret_values.max()),
                "acceptance_rate": (
                    float(np.mean(offered_probabilities)) if offered_probabilities else 1.0
                ),
                "beneficial_diversion_count": int(diversions),
                "route_entropy": _entropy(best["assignment"]),
                "safety_risk": float(best["safety"]),
                "alignment_region": region,
            }
        )
    costs = np.asarray([point["minimum_feasible_ttt"] for point in raw_points], dtype=float)
    marginal = -np.gradient(costs, epsilon) if len(epsilon) > 1 else np.zeros(1)
    points = tuple(
        AlignmentFrontierPoint(
            **point,
            marginal_cost_reduction_per_epsilon=float(marginal[index]),
        )
        for index, point in enumerate(raw_points)
    )
    knee_index, knee_epsilon = _knee(epsilon, costs)

    opportunity_count = 0
    slack_mass = 0.0
    weighted_opportunity = 0.0
    maximum_bound = float(epsilon.max())
    for user in users:
        for route_id in candidates[user.user_id]:
            if route_id == private_assignment[user.user_id]:
                continue
            slack = slacks[user.user_id][route_id]
            if slack > maximum_bound + 1e-10:
                continue
            opportunity_count += 1
            slack_mass += max(0.0, maximum_bound - slack)
            trial_assignment = {**private_assignment, user.user_id: route_id}
            trial_ttt = _ttt(network, _flows(network, routes, trial_assignment, vehicle_flow))
            weighted_opportunity += max(0.0, private["ttt"] - trial_ttt)
    return AlignmentFrontierResult(
        points=points,
        unconstrained_system_optimum_ttt=float(system_optimum["ttt"]),
        private_best_ttt=float(private["ttt"]),
        eta_only_ttt=float(eta_ttt),
        knee_epsilon=knee_epsilon,
        knee_index=knee_index,
        monotonic=bool(np.all(np.diff(costs) <= 1e-8)),
        h1_robustness={
            "binary_opportunity_count": float(opportunity_count),
            "slack_mass": float(slack_mass),
            "weighted_alignment_opportunity": float(weighted_opportunity),
            "maximum_attainable_ttt_reduction": float(private["ttt"] - costs[-1]),
            "route_diversity": float(final_route_diversity),
        },
    )
