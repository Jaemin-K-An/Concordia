from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from concordia.behavior import AcceptanceModel
from concordia.errors import ValidationError
from concordia.models import EdgeKey, Route, User
from concordia.network import RoadNetwork
from concordia.optimization.objective import total_travel_time
from concordia.preferences import UtilityModel, preference_slack


@dataclass(frozen=True)
class FixedPointIterationResult:
    value: Tuple[float, ...]
    converged: bool
    iterations: int
    final_residual: float
    solve_time_seconds: float
    residual_history: Tuple[float, ...]


def solve_fixed_point(
    initial: Sequence[float],
    mapping: Callable[[np.ndarray], Sequence[float]],
    *,
    relaxation: float = 0.5,
    tolerance: float = 1e-3,
    max_iterations: int = 50,
) -> FixedPointIterationResult:
    """Solve ``x = mapping(x)`` with relaxed Picard iteration.

    Non-convergence is an explicit result rather than an implicit last-iterate success.
    """
    if not 0 < relaxation <= 1 or tolerance <= 0 or max_iterations < 1:
        raise ValidationError("fixed-point solver parameters are invalid")
    current = np.asarray(initial, dtype=float)
    if current.ndim != 1 or not len(current) or not np.all(np.isfinite(current)):
        raise ValidationError("fixed-point initial state must be a finite vector")
    started = time.perf_counter()
    history = []
    converged = False
    for iteration in range(1, max_iterations + 1):
        mapped = np.asarray(mapping(current.copy()), dtype=float)
        if mapped.shape != current.shape or not np.all(np.isfinite(mapped)):
            raise ValidationError("fixed-point mapping returned an invalid vector")
        updated = (1.0 - relaxation) * current + relaxation * mapped
        residual = float(np.max(np.abs(updated - current)))
        history.append(residual)
        current = updated
        if residual < tolerance:
            converged = True
            break
    return FixedPointIterationResult(
        value=tuple(float(value) for value in current),
        converged=converged,
        iterations=iteration,
        final_residual=history[-1],
        solve_time_seconds=time.perf_counter() - started,
        residual_history=tuple(history),
    )


@dataclass(frozen=True)
class AcceptanceTrafficFixedPointResult:
    expected_flows: Mapping[EdgeKey, float]
    acceptance_probabilities: Mapping[str, float]
    dynamic_regrets: Mapping[str, float]
    converged: bool
    iterations: int
    final_residual: float
    solve_time_seconds: float
    residual_history: Tuple[float, ...]


class AcceptanceTrafficFixedPointSolver:
    """Couple predicted route attributes, offer acceptance, and expected edge flow."""

    def __init__(
        self,
        network: RoadNetwork,
        routes: Mapping[str, Route],
        vehicle_flow: float,
        acceptance_model: AcceptanceModel,
        *,
        relaxation: float = 0.5,
        tolerance: float = 1e-2,
        max_iterations: int = 30,
    ) -> None:
        if vehicle_flow <= 0:
            raise ValidationError("fixed-point vehicle flow must be positive")
        self.network = network
        self.routes = dict(routes)
        self.vehicle_flow = vehicle_flow
        self.acceptance_model = acceptance_model
        self.utility_model = UtilityModel()
        self.relaxation = relaxation
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.edge_order = tuple(network.edges)

    def _flows(self, assignments: Mapping[str, str]) -> Dict[EdgeKey, float]:
        flows = {edge: 0.0 for edge in self.edge_order}
        for route_id in assignments.values():
            for edge in self.routes[route_id].edges:
                flows[edge] += self.vehicle_flow
        return flows

    def _acceptance_and_regret(
        self,
        flow_guess: Mapping[EdgeKey, float],
        users: Sequence[User],
        candidates: Mapping[str, Sequence[str]],
        current_assignments: Mapping[str, str],
        proposed_assignments: Mapping[str, str],
    ) -> tuple[Dict[str, float], Dict[str, float]]:
        dynamic_routes = {
            route_id: Route(
                route_id,
                route.nodes,
                self.network.path_features(route.nodes, flow_guess),
            )
            for route_id, route in self.routes.items()
        }
        full_target = self._flows(proposed_assignments)
        current_ttt = total_travel_time(self.network, flow_guess)
        target_ttt = total_travel_time(self.network, full_target)
        acceptance: Dict[str, float] = {}
        regrets: Dict[str, float] = {}
        for user in users:
            route_ids = candidates[user.user_id]
            utilities = self.utility_model.utilities(
                user.preferences, (dynamic_routes[route_id] for route_id in route_ids)
            )
            slacks = preference_slack(utilities)
            selected_id = proposed_assignments[user.user_id]
            current_id = current_assignments[user.user_id]
            regrets[user.user_id] = slacks[selected_id]
            if selected_id == current_id:
                acceptance[user.user_id] = 1.0
                continue
            selected = dynamic_routes[selected_id].features
            current = dynamic_routes[current_id].features
            acceptance[user.user_id] = self.acceptance_model.probability(
                preference_slack=slacks[selected_id],
                utility_gain=utilities[selected_id] - utilities[current_id],
                eta_gain_minutes=current.time - selected.time,
                reliability_gain_minutes2=current.variability - selected.variability,
                network_benefit=max(0.0, current_ttt - target_ttt),
            )
        return acceptance, regrets

    def _expected_flows(
        self,
        current_assignments: Mapping[str, str],
        proposed_assignments: Mapping[str, str],
        acceptance: Mapping[str, float],
    ) -> Dict[EdgeKey, float]:
        flows = {edge: 0.0 for edge in self.edge_order}
        for user_id, proposed_route in proposed_assignments.items():
            current_route = current_assignments[user_id]
            probability = acceptance[user_id]
            for edge in self.routes[proposed_route].edges:
                flows[edge] += self.vehicle_flow * probability
            for edge in self.routes[current_route].edges:
                flows[edge] += self.vehicle_flow * (1.0 - probability)
        return flows

    def solve(
        self,
        initial_flows: Mapping[EdgeKey, float],
        users: Sequence[User],
        candidates: Mapping[str, Sequence[str]],
        current_assignments: Mapping[str, str],
        proposed_assignments: Mapping[str, str],
        *,
        max_iterations: Optional[int] = None,
    ) -> AcceptanceTrafficFixedPointResult:
        initial = [float(initial_flows.get(edge, 0.0)) for edge in self.edge_order]

        def mapping(values: np.ndarray) -> Sequence[float]:
            guess = dict(zip(self.edge_order, values.tolist()))
            acceptance, _ = self._acceptance_and_regret(
                guess,
                users,
                candidates,
                current_assignments,
                proposed_assignments,
            )
            expected = self._expected_flows(
                current_assignments, proposed_assignments, acceptance
            )
            return [expected[edge] for edge in self.edge_order]

        solved = solve_fixed_point(
            initial,
            mapping,
            relaxation=self.relaxation,
            tolerance=self.tolerance,
            max_iterations=max_iterations or self.max_iterations,
        )
        flows = dict(zip(self.edge_order, solved.value))
        acceptance, regrets = self._acceptance_and_regret(
            flows,
            users,
            candidates,
            current_assignments,
            proposed_assignments,
        )
        return AcceptanceTrafficFixedPointResult(
            expected_flows=flows,
            acceptance_probabilities=acceptance,
            dynamic_regrets=regrets,
            converged=solved.converged,
            iterations=solved.iterations,
            final_residual=solved.final_residual,
            solve_time_seconds=solved.solve_time_seconds,
            residual_history=solved.residual_history,
        )
