from __future__ import annotations

from typing import Sequence

import numpy as np


def empirical_rank_percentile(scores: Sequence[float], reference: Sequence[float]) -> np.ndarray:
    """Map scores to a fixed development reference CDF (no outcome information)."""
    values = np.sort(np.asarray(reference, dtype=float))
    if not len(values):
        raise ValueError("rank-percentile reference cannot be empty")
    return np.searchsorted(values, np.asarray(scores, dtype=float), side="right") / len(values)


def traffic_screen_recall(
    safe_success: Sequence[bool], percentiles: Sequence[float], cutoff: float
) -> float:
    success = np.asarray(safe_success, dtype=bool)
    retained = np.asarray(percentiles, dtype=float) >= cutoff
    return float((success & retained).sum() / success.sum()) if success.any() else 0.0

