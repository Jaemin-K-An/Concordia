from __future__ import annotations

import math
from itertools import combinations
from typing import Mapping, Sequence

import networkx as nx
import numpy as np

from concordia.feasibility.alignment_potential import AlignmentPotential
from concordia.models import FEATURE_NAMES, Route, User
from concordia.network import RoadNetwork


FEATURE_SCHEMA = (
    "demand",
    "volume_capacity_ratio",
    "price_of_anarchy",
    "route_overlap",
    "edge_disjointness",
    "alternative_capacity_ratio",
    "bottleneck_centrality",
    "cut_sensitivity",
    "route_diversity",
    "route_attribute_diversity",
    "preference_variance",
    "preference_entropy",
    "preference_p10",
    "preference_p90",
    "preference_bimodality",
    "preference_long_tail",
    "mean_preference_time",
    "mean_preference_variability",
    "mean_preference_cost",
    "mean_preference_risk",
    "mean_preference_complexity",
    "mean_preference_familiarity",
    "slack_mass",
    "alignment_potential_score",
    "alignment_opportunity_mass",
    "alignment_opportunity_count",
    "maximum_single_benefit",
    "acceptance_probability",
    "safety_margin",
    "phantom_risk",
    "navigation_penetration",
    "heterogeneity_rad_interaction",
)


def _jaccard(left: Route, right: Route) -> float:
    left_edges = set(left.edges)
    right_edges = set(right.edges)
    return len(left_edges & right_edges) / max(1, len(left_edges | right_edges))


def _preference_statistics(users: Sequence[User]) -> dict[str, float]:
    values = np.asarray(
        [
            [getattr(user.preferences.normalized(), name) for name in FEATURE_NAMES]
            for user in users
        ],
        dtype=float,
    )
    flattened = values.ravel()
    entropy = float(-np.mean(np.sum(values * np.log(values + 1e-12), axis=1)))
    centered = values[:, 0] - values[:, 0].mean()
    spread = max(float(values[:, 0].std()), 1e-12)
    bimodality = float(np.mean(np.abs(centered) > spread) >= 0.4 and len(users) >= 4)
    long_tail = float(np.percentile(flattened, 99) > 3.0 * np.median(flattened))
    result = {
        "preference_variance": float(np.var(values)),
        "preference_entropy": entropy,
        "preference_p10": float(np.percentile(flattened, 10)),
        "preference_p90": float(np.percentile(flattened, 90)),
        "preference_bimodality": bimodality,
        "preference_long_tail": long_tail,
    }
    result.update(
        {f"mean_preference_{name}": float(values[:, index].mean()) for index, name in enumerate(FEATURE_NAMES)}
    )
    return result


def _topology_statistics(
    network: RoadNetwork, routes: Sequence[Route], demand: float
) -> dict[str, float]:
    pairs = list(combinations(routes, 2))
    overlaps = [_jaccard(left, right) for left, right in pairs]
    primary = min(routes, key=lambda route: route.features.time)
    primary_capacity = min(network.edge_data(edge).capacity for edge in primary.edges)
    alternative_capacities = [
        min(network.edge_data(edge).capacity for edge in route.edges)
        for route in routes
        if route.route_id != primary.route_id
    ]
    betweenness = nx.edge_betweenness_centrality(network.legal_graph(), normalized=True)
    edge_connectivity = nx.edge_connectivity(
        network.legal_graph(), primary.nodes[0], primary.nodes[-1]
    )
    route_matrix = np.asarray(
        [[route.features.as_dict()[name] for name in FEATURE_NAMES] for route in routes],
        dtype=float,
    )
    scales = np.maximum(np.mean(np.abs(route_matrix), axis=0), 1e-9)
    normalized = route_matrix / scales
    route_attribute_diversity = (
        float(np.trace(np.cov(normalized, rowvar=False))) if len(routes) > 1 else 0.0
    )
    unique_capacity = sum(
        network.edge_data(edge).capacity
        for edge in set().union(*(set(route.edges) for route in routes))
    )
    return {
        "volume_capacity_ratio": float(demand / max(unique_capacity, 1e-12)),
        "route_overlap": float(np.mean(overlaps)) if overlaps else 1.0,
        "edge_disjointness": 1.0 - (float(np.mean(overlaps)) if overlaps else 1.0),
        "alternative_capacity_ratio": float(
            sum(alternative_capacities) / max(primary_capacity, 1e-12)
        ),
        "bottleneck_centrality": float(max(betweenness.values(), default=0.0)),
        "cut_sensitivity": 1.0 / max(float(edge_connectivity), 1.0),
        "route_diversity": float(np.mean([1.0 - value for value in overlaps])) if overlaps else 0.0,
        "route_attribute_diversity": route_attribute_diversity,
    }


def extract_feasibility_features(
    *,
    network: RoadNetwork,
    routes: Mapping[str, Route],
    users: Sequence[User],
    demand: float,
    price_of_anarchy: float,
    alignment: AlignmentPotential,
    acceptance_probability: float,
    safety_margin: float,
    navigation_penetration: float,
) -> dict[str, float]:
    topology = _topology_statistics(network, list(routes.values()), demand)
    preference = _preference_statistics(users)
    phantom_risk = 1.0 / (
        1.0
        + math.exp(
            -4.0
            * (
                topology["volume_capacity_ratio"]
                + 0.35 * topology["route_overlap"]
                - 0.75
            )
        )
    )
    features = {
        "demand": float(demand),
        "price_of_anarchy": float(price_of_anarchy),
        **topology,
        **preference,
        "slack_mass": alignment.slack_mass,
        "alignment_potential_score": alignment.score,
        "alignment_opportunity_mass": alignment.opportunity_mass,
        "alignment_opportunity_count": float(alignment.opportunity_count),
        "maximum_single_benefit": alignment.maximum_single_benefit,
        "acceptance_probability": float(acceptance_probability),
        "safety_margin": float(safety_margin),
        "phantom_risk": float(phantom_risk),
        "navigation_penetration": float(navigation_penetration),
        "heterogeneity_rad_interaction": preference["preference_variance"]
        * topology["route_attribute_diversity"],
    }
    return {name: float(features[name]) for name in FEATURE_SCHEMA}
