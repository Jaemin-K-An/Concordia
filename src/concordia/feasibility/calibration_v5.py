from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from concordia.feasibility.calibration_v4 import ProbabilityCalibrator, calibration_error


def unified_calibration_metrics(
    labels: Sequence[int], probability: Sequence[float], *, bins: int = 10
) -> dict[str, Any]:
    if bins != 10:
        raise ValueError("v5 calibration protocol is frozen to 10 equal-width bins")
    metrics = calibration_error(labels, probability)
    metrics.update(
        {
            "protocol": "equal-width bins on [0,1]; empty bins omitted",
            "bin_count": bins,
            "ece_definition": "sum_b (n_b/N) * abs(mean_probability_b-observed_rate_b)",
            "evaluation_only": True,
        }
    )
    return metrics


@dataclass
class RegimeProbabilityCalibrator:
    method: str
    global_calibrator: ProbabilityCalibrator
    regime_calibrators: dict[str, ProbabilityCalibrator]

    @classmethod
    def fit(
        cls,
        method: str,
        probability: Sequence[float],
        labels: Sequence[int],
        regimes: Sequence[str],
        *,
        minimum_regime_size: int = 30,
    ) -> "RegimeProbabilityCalibrator":
        raw = np.asarray(probability, dtype=float)
        y = np.asarray(labels, dtype=int)
        values = np.asarray(regimes, dtype=object)
        global_calibrator = ProbabilityCalibrator(method).fit(raw, y)
        fitted = {}
        for regime in sorted(set(regimes)):
            mask = values == regime
            if int(mask.sum()) < minimum_regime_size or len(np.unique(y[mask])) < 2:
                continue
            fitted[regime] = ProbabilityCalibrator(method).fit(raw[mask], y[mask])
        return cls(method, global_calibrator, fitted)

    def predict(self, probability: Sequence[float], regimes: Sequence[str]) -> np.ndarray:
        raw = np.asarray(probability, dtype=float)
        output = self.global_calibrator.predict(raw)
        values = np.asarray(regimes, dtype=object)
        for regime, calibrator in self.regime_calibrators.items():
            mask = values == regime
            output[mask] = calibrator.predict(raw[mask])
        return np.clip(output, 0.0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "global_calibrator": self.global_calibrator.to_dict(),
            "regime_calibrators": {
                regime: calibrator.to_dict()
                for regime, calibrator in self.regime_calibrators.items()
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RegimeProbabilityCalibrator":
        return cls(
            str(value["method"]),
            ProbabilityCalibrator.from_dict(value["global_calibrator"]),
            {
                str(regime): ProbabilityCalibrator.from_dict(package)
                for regime, package in value["regime_calibrators"].items()
            },
        )
