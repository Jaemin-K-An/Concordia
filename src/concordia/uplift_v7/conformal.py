from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


def _higher_quantile(values: np.ndarray, quantile: float) -> float:
    try:
        return float(np.quantile(values, quantile, method="higher"))
    except TypeError:
        return float(np.quantile(values, quantile, interpolation="higher"))


@dataclass(frozen=True)
class ConformalAdjustments:
    miscoverage: float
    traffic_radius: float
    safety_upper_adjustment: float
    regret_upper_adjustment: float

    @classmethod
    def fit(
        cls,
        miscoverage: float,
        traffic_actual: Sequence[float],
        traffic_prediction: Sequence[float],
        safety_actual: Sequence[float],
        safety_prediction: Sequence[float],
        regret_actual: Sequence[float],
        regret_prediction: Sequence[float],
    ) -> "ConformalAdjustments":
        quantile = min(1.0, (1.0 - miscoverage) * (len(traffic_actual) + 1) / len(traffic_actual))
        traffic_error = np.abs(np.asarray(traffic_actual) - np.asarray(traffic_prediction))
        safety_error = np.asarray(safety_actual) - np.asarray(safety_prediction)
        regret_error = np.asarray(regret_actual) - np.asarray(regret_prediction)
        return cls(
            float(miscoverage),
            _higher_quantile(traffic_error, quantile),
            max(0.0, _higher_quantile(safety_error, quantile)),
            max(0.0, _higher_quantile(regret_error, quantile)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "miscoverage": self.miscoverage,
            "traffic_radius": self.traffic_radius,
            "safety_upper_adjustment": self.safety_upper_adjustment,
            "regret_upper_adjustment": self.regret_upper_adjustment,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ConformalAdjustments":
        return cls(
            float(value["miscoverage"]), float(value["traffic_radius"]),
            float(value["safety_upper_adjustment"]),
            float(value["regret_upper_adjustment"]),
        )

