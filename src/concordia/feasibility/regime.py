from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class RegimeDefinition:
    low_penetration_boundary: float
    high_penetration_boundary: float
    high_overlap_boundary: float
    low_alternative_capacity_boundary: float
    discovery_score: float

    @classmethod
    def discover(
        cls,
        rows: Sequence[Mapping[str, Any]],
        labels: Sequence[int],
        *,
        cut_candidates: Sequence[float],
        minimum_size: int,
    ) -> "RegimeDefinition":
        penetration = np.asarray(
            [float(row["condition"]["navigation_penetration"]) for row in rows]
        )
        y = np.asarray(labels, dtype=float)
        best: tuple[float, float, float] | None = None
        for low, high in combinations(sorted(set(float(v) for v in cut_candidates)), 2):
            groups = (penetration < low, (penetration >= low) & (penetration < high), penetration >= high)
            if min(int(mask.sum()) for mask in groups) < minimum_size:
                continue
            residual = sum(float(((y[mask] - y[mask].mean()) ** 2).sum()) for mask in groups)
            between = float(np.var([y[mask].mean() for mask in groups]))
            score = between - residual / max(1, len(y))
            candidate = (score, low, high)
            if best is None or candidate > best:
                best = candidate
        if best is None:
            best = (0.0, 0.40, 0.80)
        overlaps = np.asarray([float(row["features"]["route_overlap"]) for row in rows])
        capacities = np.asarray(
            [float(row["features"]["alternative_capacity_ratio"]) for row in rows]
        )
        return cls(
            low_penetration_boundary=float(best[1]),
            high_penetration_boundary=float(best[2]),
            high_overlap_boundary=float(np.quantile(overlaps, 0.75)),
            low_alternative_capacity_boundary=float(np.quantile(capacities, 0.25)),
            discovery_score=float(best[0]),
        )

    def route(self, features: Mapping[str, float]) -> str:
        penetration = float(features["navigation_penetration"])
        constrained = (
            float(features["route_overlap"]) >= self.high_overlap_boundary
            and float(features["alternative_capacity_ratio"])
            <= self.low_alternative_capacity_boundary
        )
        if constrained:
            return "STRUCTURALLY_CONSTRAINED"
        if penetration >= self.high_penetration_boundary:
            return "HIGH_CONTROL"
        if penetration >= self.low_penetration_boundary:
            return "PARTIAL_CONTROL"
        return "LOW_CONTROL"

    def to_dict(self) -> dict[str, float]:
        return {
            "low_penetration_boundary": self.low_penetration_boundary,
            "high_penetration_boundary": self.high_penetration_boundary,
            "high_overlap_boundary": self.high_overlap_boundary,
            "low_alternative_capacity_boundary": self.low_alternative_capacity_boundary,
            "discovery_score": self.discovery_score,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RegimeDefinition":
        return cls(**{key: float(value[key]) for key in cls.__dataclass_fields__})
