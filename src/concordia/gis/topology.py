from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

import networkx as nx

from concordia.errors import ValidationError
from concordia.models import EdgeKey
from concordia.network import RoadNetwork


@dataclass(frozen=True)
class TopologyAudit:
    node_count: int
    edge_count: int
    weak_component_count: int
    largest_component_fraction: float
    alternative_route_count: int
    valid: bool
    reasons: Tuple[str, ...]


def largest_weak_component(
    network: RoadNetwork,
    coordinates: Mapping[str, Tuple[float, float]],
    geometries: Mapping[EdgeKey, Tuple[Tuple[float, float], ...]],
) -> Tuple[RoadNetwork, Dict[str, Tuple[float, float]], Dict[EdgeKey, Tuple[Tuple[float, float], ...]]]:
    """Return a copied, routable largest weak component without modifying raw topology."""
    components = list(nx.weakly_connected_components(network.graph))
    if not components:
        raise ValidationError("cannot select a component from an empty network")
    selected_nodes = max(components, key=lambda component: (len(component), sorted(component)[0]))
    selected = RoadNetwork(f"{network.name}_largest_component")
    selected_geometries = {}
    for edge in network.edges:
        if edge[0] in selected_nodes and edge[1] in selected_nodes:
            selected.add_edge(edge[0], edge[1], network.edge_data(edge))
            selected_geometries[edge] = geometries[edge]
    return (
        selected,
        {node: coordinates[node] for node in selected_nodes},
        selected_geometries,
    )


def audit_topology(
    network: RoadNetwork,
    origin: str,
    destination: str,
    minimum_alternative_routes: int = 2,
) -> TopologyAudit:
    if minimum_alternative_routes < 1:
        raise ValidationError("minimum alternative route count must be positive")
    components = list(nx.weakly_connected_components(network.graph))
    largest = max((len(component) for component in components), default=0)
    reasons = []
    try:
        routes = network.candidate_paths(origin, destination, k=minimum_alternative_routes, max_overlap=1)
    except ValidationError:
        routes = []
        reasons.append("disconnected_or_missing_od")
    if len(routes) < minimum_alternative_routes:
        reasons.append("insufficient_route_diversity")
    if len(components) > 1:
        reasons.append("multiple_weak_components")
    return TopologyAudit(
        node_count=network.graph.number_of_nodes(),
        edge_count=network.graph.number_of_edges(),
        weak_component_count=len(components),
        largest_component_fraction=largest / max(1, network.graph.number_of_nodes()),
        alternative_route_count=len(routes),
        valid=not reasons,
        reasons=tuple(reasons),
    )
