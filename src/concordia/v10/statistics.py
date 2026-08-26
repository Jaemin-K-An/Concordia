from __future__ import annotations

from typing import Sequence

import numpy as np


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return 0.0, 0.0
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def robust_mean(values: Sequence[float], penalty: float) -> float:
    mean, standard_deviation = mean_std(values)
    return float(mean - penalty * standard_deviation)


def empirical_lcb(values: Sequence[float], quantile: float = 0.10) -> float:
    if not 0.0 <= quantile <= 1.0 or not values:
        raise ValueError("empirical LCB requires values and a valid quantile")
    return float(np.quantile(np.asarray(values, dtype=float), quantile, method="linear"))

