from __future__ import annotations

import math
from typing import Dict, Sequence

import numpy as np

from concordia.errors import ValidationError


def summarize_samples(values: Sequence[float], seed: int = 0, bootstrap_samples: int = 2000) -> Dict[str, float]:
    if not values or bootstrap_samples < 100:
        raise ValidationError("statistics require values and at least 100 bootstrap samples")
    data = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(data)):
        raise ValidationError("statistics input contains NaN or infinity")
    rng = np.random.default_rng(seed)
    draws = rng.choice(data, size=(bootstrap_samples, len(data)), replace=True).mean(axis=1)
    return {
        "count": int(len(data)),
        "mean": float(data.mean()),
        "std": float(data.std(ddof=1)) if len(data) > 1 else 0.0,
        "median": float(np.median(data)),
        "p90": float(np.percentile(data, 90)),
        "p95": float(np.percentile(data, 95)),
        "max": float(data.max()),
        "ci95_low": float(np.percentile(draws, 2.5)),
        "ci95_high": float(np.percentile(draws, 97.5)),
    }


def paired_effect_size(baseline: Sequence[float], proposed: Sequence[float]) -> float:
    if len(baseline) != len(proposed) or len(baseline) < 2:
        raise ValidationError("paired effect size needs matched samples of length >= 2")
    differences = np.asarray(proposed, dtype=float) - np.asarray(baseline, dtype=float)
    std = float(differences.std(ddof=1))
    if math.isclose(std, 0.0):
        return 0.0 if math.isclose(float(differences.mean()), 0.0) else math.copysign(float("inf"), float(differences.mean()))
    return float(differences.mean() / std)
