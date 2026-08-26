from __future__ import annotations

from typing import Sequence

import numpy as np


def robust_benefit(mean_benefit: float, standard_deviation: float, penalty: float) -> float:
    return float(mean_benefit - penalty * max(standard_deviation, 0.0))


def rollout_summary(traffic: Sequence[float], unsafe: Sequence[bool]) -> dict:
    benefit = np.asarray(traffic, dtype=float)
    risk = np.asarray(unsafe, dtype=bool)
    return {
        "mean_traffic_benefit": float(benefit.mean()) if len(benefit) else 0.0,
        "traffic_benefit_std": float(benefit.std(ddof=1)) if len(benefit) > 1 else 0.0,
        "rollout_unsafe_probability": float(risk.mean()) if len(risk) else 1.0,
        "rollout_count": len(benefit),
    }

