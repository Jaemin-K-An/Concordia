from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Tuple

import networkx as nx

from concordia.errors import ValidationError
from concordia.models import EdgeKey
from concordia.network import RoadNetwork


@dataclass(frozen=True)
class TrafficAssignmentResult:
    flows: Mapping[EdgeKey, float]
    total_travel_time: float
    objective: float
    relative_gap: float
    iterations: int
    converged: bool


class TrafficAssignment:
    """Deterministic Frank-Wolfe UE and SO for separable BPR link costs."""

    def __init__(self, network: RoadNetwork) -> None:
        self.network = network

    def _validate_demand(self, demand: Mapping[Tuple[str, str], float]) -> None:
        if not demand:
            raise ValidationError("traffic assignment requires at least one OD demand")
        for (origin, destination), value in demand.items():
            if value <= 0:
                raise ValidationError("OD demand must be positive")
            self.network.validate_od(origin, destination)

    def _edge_costs(self, flows: Mapping[EdgeKey, float], system_optimal: bool) -> Dict[EdgeKey, float]:
        result: Dict[EdgeKey, float] = {}
        for edge in self.network.edges:
            data = self.network.edge_data(edge)
            flow = float(flows.get(edge, 0.0))
            cost = data.travel_time(flow)
            if system_optimal:
                cost += flow * data.derivative(flow)
            result[edge] = cost
        return result

    def _all_or_nothing(
        self,
        demand: Mapping[Tuple[str, str], float],
        costs: Mapping[EdgeKey, float],
    ) -> Dict[EdgeKey, float]:
        graph = self.network.legal_graph()
        for edge in graph.edges:
            graph.edges[edge]["assignment_cost"] = float(costs[edge])
        flows = {edge: 0.0 for edge in self.network.edges}
        for (origin, destination), volume in sorted(demand.items()):
            try:
                path = nx.shortest_path(graph, origin, destination, weight="assignment_cost")
            except nx.NetworkXNoPath as exc:
                raise ValidationError(f"no legal assignment path for {origin}->{destination}") from exc
            for edge in zip(path, path[1:]):
                flows[edge] += float(volume)
        return flows

    def total_travel_time(self, flows: Mapping[EdgeKey, float]) -> float:
        return sum(
            float(flows.get(edge, 0.0))
            * self.network.edge_data(edge).travel_time(float(flows.get(edge, 0.0)))
            for edge in self.network.edges
        )

    def beckmann_objective(self, flows: Mapping[EdgeKey, float]) -> float:
        return sum(
            self.network.edge_data(edge).integral(float(flows.get(edge, 0.0)))
            for edge in self.network.edges
        )

    @staticmethod
    def _blend(
        current: Mapping[EdgeKey, float], target: Mapping[EdgeKey, float], step: float
    ) -> Dict[EdgeKey, float]:
        return {
            edge: float(current[edge]) + step * (float(target[edge]) - float(current[edge]))
            for edge in current
        }

    def _line_search(
        self,
        current: Mapping[EdgeKey, float],
        target: Mapping[EdgeKey, float],
        objective: Callable[[Mapping[EdgeKey, float]], float],
    ) -> float:
        """Golden-section search on the convex Frank-Wolfe segment."""
        left, right = 0.0, 1.0
        ratio = (5**0.5 - 1) / 2
        x1 = right - ratio * (right - left)
        x2 = left + ratio * (right - left)
        f1 = objective(self._blend(current, target, x1))
        f2 = objective(self._blend(current, target, x2))
        for _ in range(64):
            if f1 <= f2:
                right, x2, f2 = x2, x1, f1
                x1 = right - ratio * (right - left)
                f1 = objective(self._blend(current, target, x1))
            else:
                left, x1, f1 = x1, x2, f2
                x2 = left + ratio * (right - left)
                f2 = objective(self._blend(current, target, x2))
        candidates = (left, (left + right) / 2, right)
        return min(candidates, key=lambda value: objective(self._blend(current, target, value)))

    def solve(
        self,
        demand: Mapping[Tuple[str, str], float],
        system_optimal: bool = False,
        tolerance: float = 1e-7,
        max_iterations: int = 500,
    ) -> TrafficAssignmentResult:
        self._validate_demand(demand)
        if tolerance <= 0 or max_iterations < 1:
            raise ValidationError("positive tolerance and max_iterations are required")
        zero = {edge: 0.0 for edge in self.network.edges}
        flows = self._all_or_nothing(demand, self._edge_costs(zero, system_optimal))
        objective = self.total_travel_time if system_optimal else self.beckmann_objective
        gap = float("inf")
        converged = False
        iteration = 0
        for iteration in range(1, max_iterations + 1):
            costs = self._edge_costs(flows, system_optimal)
            target = self._all_or_nothing(demand, costs)
            current_cost = sum(float(flows[edge]) * costs[edge] for edge in flows)
            shortest_cost = sum(float(target[edge]) * costs[edge] for edge in target)
            gap = max(0.0, current_cost - shortest_cost) / max(abs(current_cost), 1e-12)
            if gap <= tolerance:
                converged = True
                break
            step = self._line_search(flows, target, objective)
            flows = self._blend(flows, target, step)
        return TrafficAssignmentResult(
            flows=flows,
            total_travel_time=self.total_travel_time(flows),
            objective=objective(flows),
            relative_gap=gap,
            iterations=iteration,
            converged=converged,
        )

    def user_equilibrium(self, demand: Mapping[Tuple[str, str], float], **kwargs: object) -> TrafficAssignmentResult:
        return self.solve(demand, system_optimal=False, **kwargs)

    def system_optimum(self, demand: Mapping[Tuple[str, str], float], **kwargs: object) -> TrafficAssignmentResult:
        return self.solve(demand, system_optimal=True, **kwargs)

    def price_of_anarchy(self, demand: Mapping[Tuple[str, str], float]) -> float:
        ue = self.user_equilibrium(demand)
        so = self.system_optimum(demand)
        if not ue.converged or not so.converged:
            raise ValidationError("PoA requires converged UE and SO assignments")
        return ue.total_travel_time / so.total_travel_time
