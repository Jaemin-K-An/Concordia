from __future__ import annotations

from typing import List, Mapping, Tuple

from concordia.models import EdgeKey
from concordia.network import RoadNetwork


def rank_hotspots(
    network: RoadNetwork,
    flows: Mapping[EdgeKey, float],
    limit: int = 10,
) -> List[Tuple[EdgeKey, float]]:
    """Rank analytical bottlenecks by saturation plus declared surrogate risk."""
    scores = []
    for edge in network.edges:
        data = network.edge_data(edge)
        saturation = float(flows.get(edge, 0.0)) / data.capacity
        scores.append((edge, saturation + data.risk))
    return sorted(scores, key=lambda item: (-item[1], item[0]))[:limit]

