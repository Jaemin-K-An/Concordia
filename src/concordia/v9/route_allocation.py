from __future__ import annotations

import math
from typing import Mapping, Sequence


def allocation_weights(
    strategy: str,
    alternatives: Sequence[Mapping[str, float]],
) -> tuple[float, ...]:
    if not alternatives:
        raise ValueError("at least one legal alternative is required")
    if not all(bool(route.get("legal", True)) for route in alternatives):
        raise ValueError("illegal route in allocation candidate")
    if strategy == "R1_best_alternative":
        best = min(range(len(alternatives)), key=lambda index: float(alternatives[index]["time"]))
        raw = [float(index == best) for index in range(len(alternatives))]
    elif strategy == "R2_capacity_proportional":
        raw = [max(float(route["capacity"]), 1e-9) for route in alternatives]
    elif strategy == "R3_inverse_overlap":
        raw = [1.0 / max(float(route["overlap"]), 0.05) for route in alternatives]
    elif strategy == "R4_reliability_weighted":
        raw = [1.0 / max(float(route["variability"]), 0.05) for route in alternatives]
    elif strategy == "R5_minimum_bottleneck_load":
        raw = [1.0 / max(float(route["bottleneck_load"]), 0.05) for route in alternatives]
    elif strategy == "R6_entropy_constrained":
        raw = [1.0 for _ in alternatives]
    else:
        raise ValueError(f"unknown route allocation: {strategy}")
    total = sum(raw)
    return tuple(value / total for value in raw)


def route_entropy(weights: Sequence[float]) -> float:
    return float(-sum(value * math.log(max(value, 1e-12)) for value in weights))


def action_concentration_index(weights: Sequence[float]) -> float:
    total = sum(abs(value) for value in weights)
    return float(sum((abs(value) / max(total, 1e-12)) ** 2 for value in weights))


def deterministic_route_index(vehicle_id: str, weights: Sequence[float]) -> int:
    token = int("".join(character for character in vehicle_id if character.isdigit()) or 0)
    point = ((token * 2654435761) % 1_000_003) / 1_000_003.0
    cumulative = 0.0
    for index, weight in enumerate(weights):
        cumulative += float(weight)
        if point <= cumulative + 1e-12:
            return index
    return len(weights) - 1

