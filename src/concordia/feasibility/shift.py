from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass
class RobustShiftDetector:
    median: np.ndarray
    scale: np.ndarray
    mild_distance: float
    strong_distance: float

    @classmethod
    def fit(
        cls,
        matrix: np.ndarray,
        *,
        mild_quantile: float = 0.90,
        strong_quantile: float = 0.99,
    ) -> "RobustShiftDetector":
        x = np.asarray(matrix, dtype=float)
        median = np.median(x, axis=0)
        mad = np.median(np.abs(x - median), axis=0) * 1.4826
        standard = x.std(axis=0)
        scale = np.where(mad > 1e-8, mad, np.where(standard > 1e-8, standard, 1.0))
        distance = np.sqrt(np.mean(((x - median) / scale) ** 2, axis=1))
        return cls(
            median,
            scale,
            float(np.quantile(distance, mild_quantile)),
            float(np.quantile(distance, strong_quantile)),
        )

    def distance(self, matrix: np.ndarray) -> np.ndarray:
        x = np.asarray(matrix, dtype=float)
        return np.sqrt(np.mean(((x - self.median) / self.scale) ** 2, axis=1))

    def score(self, matrix: np.ndarray) -> np.ndarray:
        return np.clip(self.distance(matrix) / max(self.strong_distance, 1e-9), 0.0, 4.0)

    def classify(self, matrix: np.ndarray) -> list[str]:
        distance = self.distance(matrix)
        return [
            "IN_DISTRIBUTION"
            if value <= self.mild_distance
            else "MILD_SHIFT"
            if value <= self.strong_distance
            else "STRONG_SHIFT"
            for value in distance
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "median": self.median.tolist(),
            "scale": self.scale.tolist(),
            "mild_distance": self.mild_distance,
            "strong_distance": self.strong_distance,
            "definition": "root mean squared robust standardized feature distance",
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RobustShiftDetector":
        return cls(
            np.asarray(value["median"], dtype=float),
            np.asarray(value["scale"], dtype=float),
            float(value["mild_distance"]),
            float(value["strong_distance"]),
        )
