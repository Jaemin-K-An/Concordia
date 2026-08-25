from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from concordia.feasibility.calibration import wilson_interval


@dataclass(frozen=True)
class ConformalRiskController:
    probability_threshold: float
    target_error_rate: float
    calibration_error_upper: float

    @classmethod
    def fit(
        cls,
        probabilities: Sequence[float],
        labels: Sequence[int],
        *,
        target_error_rate: float = 0.20,
        thresholds: Sequence[float],
    ) -> "ConformalRiskController":
        p = np.asarray(probabilities, dtype=float)
        y = np.asarray(labels, dtype=int)
        candidates = []
        for threshold in sorted(set(float(item) for item in thresholds)):
            selected = p >= threshold
            count = int(selected.sum())
            if not count:
                continue
            errors = int(np.sum(y[selected] == 0))
            precision_lower, _upper = wilson_interval(count - errors, count)
            error_upper = 1.0 - precision_lower
            candidates.append((threshold, count, error_upper))
        if not candidates:
            raise ValueError("conformal calibration found no selectable probability threshold")
        valid = [row for row in candidates if row[2] <= target_error_rate]
        selected = max(valid, key=lambda row: row[1]) if valid else min(
            candidates, key=lambda row: (row[2], -row[1])
        )
        return cls(selected[0], target_error_rate, selected[2])

    def intervene(self, probabilities: Sequence[float]) -> np.ndarray:
        return np.asarray(probabilities, dtype=float) >= self.probability_threshold

    def to_dict(self) -> dict:
        return {
            "probability_threshold": self.probability_threshold,
            "target_error_rate": self.target_error_rate,
            "calibration_error_upper": self.calibration_error_upper,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, float]) -> "ConformalRiskController":
        return cls(
            float(value["probability_threshold"]),
            float(value["target_error_rate"]),
            float(value["calibration_error_upper"]),
        )
