from __future__ import annotations

import math
from collections import Counter
from typing import Mapping, Sequence

import numpy as np

from concordia.errors import ValidationError
from concordia.models import EdgeKey
from concordia.network import RoadNetwork


def total_travel_time(network: RoadNetwork, flows: Mapping[EdgeKey, float]) -> float:
    return sum(
        float(flows.get(edge, 0.0))
        * network.edge_data(edge).travel_time(float(flows.get(edge, 0.0)))
        for edge in network.edges
    )


def route_concentration(assignments: Mapping[str, str]) -> tuple[float, float]:
    if not assignments:
        raise ValidationError("route concentration requires assignments")
    counts = Counter(assignments.values())
    shares = [count / len(assignments) for count in counts.values()]
    hhi = sum(share**2 for share in shares)
    entropy = -sum(share * math.log(share) for share in shares if share > 0)
    return hhi, entropy


def upper_cvar(values: Sequence[float], alpha: float = 0.95) -> float:
    if not values or not 0 <= alpha < 1:
        raise ValidationError("CVaR requires values and alpha in [0, 1)")
    data = np.sort(np.asarray(values, dtype=float))
    if not np.all(np.isfinite(data)) or np.any(data < 0):
        raise ValidationError("risk values must be finite and non-negative")
    start = min(len(data) - 1, int(math.floor(alpha * len(data))))
    return float(data[start:].mean())

