from __future__ import annotations

from typing import Sequence

import numpy as np

from concordia.feasibility.calibration_v4 import ProbabilityCalibrator, calibration_error


def calibration_diagnostics(labels: Sequence[int], probabilities: Sequence[float]) -> dict:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    output = calibration_error(y, p)
    for name, mask in (
        ("low_probability_below_0_05", p < 0.05),
        ("highest_risk_20_percent", p >= np.quantile(p, 0.80)),
        ("unsafe_class", y == 1),
    ):
        output[name] = {
            "count": int(mask.sum()),
            "mean_probability": float(p[mask].mean()) if mask.any() else None,
            "observed_unsafe_rate": float(y[mask].mean()) if mask.any() else None,
            "brier_score": float(np.mean((p[mask] - y[mask]) ** 2)) if mask.any() else None,
        }
    return output


__all__ = ["ProbabilityCalibrator", "calibration_diagnostics"]

