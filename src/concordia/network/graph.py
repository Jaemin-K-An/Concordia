from __future__ import annotations

from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import networkx as nx

from concordia.errors import ValidationError
from concordia.models import EdgeData, EdgeKey, Route, RouteFeatures


class RoadNetwork:
    def __init__(self, name: str = "network") -> None:
        self.name = name
        self.graph = nx.DiGraph(name=name)

    def add_edge(self, source: str, target: str, data: EdgeData) -> None:
        if source == target:
            raise ValidationError("self-loop road edges are not supported")
        self.graph.add_edge(source, target, data=data)

    def edge_data(self, edge: EdgeKey) -> EdgeData:
        try:
            return self.graph.edges[edge]["data"]
        except KeyError as exc:
            raise ValidationError(f"unknown edge {edge}") from exc

    def validate_od(self, origin: str, destination: str) -> None:
        if origin not in self.graph or destination not in self.graph:
            raise ValidationError(f"OD nodes {origin}->{destination} are not in the network")
        if not nx.has_path(self.legal_graph(), origin, destination):
            raise ValidationError(f"no legal path for OD {origin}->{destination}")

    def legal_graph(self) -> nx.DiGraph:
        legal = nx.DiGraph(name=self.name)
        for source, target, attrs in self.graph.edges(data=True):
            if attrs["data"].legal:
                legal.add_edge(source, target, **attrs)
        return legal

    def path_edges(self, nodes: Sequence[str]) -> Tuple[EdgeKey, ...]:
        if len(nodes) < 2:
            raise ValidationError("path must contain at least two nodes")
        edges = tuple(zip(nodes, nodes[1:]))
        for edge in edges:
            data = self.edge_data(edge)
            if not data.legal:
                raise ValidationError(f"path uses illegal edge {edge}")
        return edges

    def path_features(
        self, nodes: Sequence[str], flows: Optional[Mapping[EdgeKey, float]] = None
    ) -> RouteFeatures:
        flows = flows or {}
        edges = self.path_edges(nodes)
        data = [self.edge_data(edge) for edge in edges]
        return RouteFeatures(
            time=sum(item.travel_time(float(flows.get(edge, 0.0))) for edge, item in zip(edges, data)),
            variability=sum(item.variability for item in data),
            cost=sum(item.monetary_cost for item in data),
            risk=sum(item.risk for item in data),
            complexity=sum(item.complexity for item in data) / len(data),
            familiarity=0.0,
        )

    def make_route(
        self,
        route_id: str,
        nodes: Sequence[str],
        flows: Optional[Mapping[EdgeKey, float]] = None,
        familiarity: float = 0.0,
    ) -> Route:
        features = self.path_features(nodes, flows)
        features = RouteFeatures(**{**features.as_dict(), "familiarity": familiarity})
        return Route(route_id=route_id, nodes=tuple(nodes), features=features)

    @staticmethod
    def overlap_coefficient(first: Sequence[str], second: Sequence[str]) -> float:
        first_edges = set(zip(first, first[1:]))
        second_edges = set(zip(second, second[1:]))
        denominator = min(len(first_edges), len(second_edges))
        return len(first_edges & second_edges) / denominator if denominator else 1.0

    def candidate_paths(
        self,
        origin: str,
        destination: str,
        k: int = 5,
        max_overlap: float = 0.85,
    ) -> List[Tuple[str, ...]]:
        self.validate_od(origin, destination)
        if k < 1 or not 0 <= max_overlap <= 1:
            raise ValidationError("k >= 1 and max_overlap in [0, 1] are required")
        graph = self.legal_graph()
        for source, target, attrs in graph.edges(data=True):
            attrs["weight"] = attrs["data"].free_flow_time
        accepted: List[Tuple[str, ...]] = []
        try:
            generator = nx.shortest_simple_paths(graph, origin, destination, weight="weight")
            for path in generator:
                candidate = tuple(path)
                if not accepted or all(
                    self.overlap_coefficient(candidate, existing) <= max_overlap
                    for existing in accepted
                ):
                    accepted.append(candidate)
                if len(accepted) >= k:
                    break
        except nx.NetworkXNoPath as exc:
            raise ValidationError(f"no candidate path for {origin}->{destination}") from exc
        if not accepted:
            raise ValidationError(f"candidate generation returned no path for {origin}->{destination}")
        return accepted

    def candidate_routes(
        self,
        origin: str,
        destination: str,
        k: int = 5,
        max_overlap: float = 0.85,
        pareto_filter: bool = False,
    ) -> List[Route]:
        routes = [
            self.make_route(f"{origin}-{destination}-{index}", nodes)
            for index, nodes in enumerate(self.candidate_paths(origin, destination, k, max_overlap))
        ]
        return self.pareto_front(routes) if pareto_filter else routes

    def multiobjective_candidate_routes(
        self,
        origin: str,
        destination: str,
        flows: Optional[Mapping[EdgeKey, float]] = None,
        k_per_objective: int = 2,
        max_overlap: float = 0.95,
        pareto_filter: bool = True,
    ) -> List[Route]:
        """Union paths generated from ETA, reliability, cost, risk, and complexity costs."""
        self.validate_od(origin, destination)
        if k_per_objective < 1:
            raise ValidationError("k_per_objective must be positive")
        flows = flows or {}
        objectives = ("eta", "reliability", "cost", "risk", "complexity")
        discovered: List[Tuple[str, ...]] = []
        graph = self.legal_graph()
        for objective in objectives:
            weight_name = f"candidate_{objective}"
            for edge in graph.edges:
                data = graph.edges[edge]["data"]
                edge_flow = float(flows.get(edge, 0.0))
                value = {
                    "eta": data.travel_time(edge_flow),
                    "reliability": data.variability,
                    "cost": data.monetary_cost,
                    "risk": data.risk,
                    "complexity": data.complexity,
                }[objective]
                # A tiny ETA tie-break makes zero-valued criteria deterministic.
                graph.edges[edge][weight_name] = float(value) + 1e-9 * data.free_flow_time
            generator = nx.shortest_simple_paths(
                graph,
                origin,
                destination,
                weight=weight_name,
            )
            accepted_for_objective = 0
            for nodes in generator:
                path = tuple(nodes)
                if path not in discovered:
                    discovered.append(path)
                accepted_for_objective += 1
                if accepted_for_objective >= k_per_objective:
                    break
        diverse: List[Tuple[str, ...]] = []
        for path in discovered:
            if not diverse or all(
                self.overlap_coefficient(path, existing) <= max_overlap for existing in diverse
            ):
                diverse.append(path)
        routes = [
            self.make_route(f"{origin}-{destination}-mo-{index}", path, flows=flows)
            for index, path in enumerate(diverse)
        ]
        if not routes:
            raise ValidationError("multi-objective generation produced no diverse route")
        return self.pareto_front(routes) if pareto_filter else routes

    @staticmethod
    def pareto_front(routes: Iterable[Route]) -> List[Route]:
        routes = list(routes)

        def dominates(left: Route, right: Route) -> bool:
            left_values = left.features.as_dict()
            right_values = right.features.as_dict()
            costs = ("time", "variability", "cost", "risk", "complexity")
            weak = all(left_values[name] <= right_values[name] for name in costs)
            weak = weak and left_values["familiarity"] >= right_values["familiarity"]
            strict = any(left_values[name] < right_values[name] for name in costs)
            strict = strict or left_values["familiarity"] > right_values["familiarity"]
            return weak and strict

        return [route for route in routes if not any(dominates(other, route) for other in routes)]

    @property
    def edges(self) -> Tuple[EdgeKey, ...]:
        return tuple(self.graph.edges())
