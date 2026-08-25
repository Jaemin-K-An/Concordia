from __future__ import annotations

from typing import Sequence

import numpy as np


def false_safe_rate(
    predicted_upper: Sequence[float], actual_effect: Sequence[float], margin: float
) -> dict[str, float | int]:
    upper = np.asarray(predicted_upper, dtype=float)
    actual = np.asarray(actual_effect, dtype=float)
    predicted_safe = upper <= margin
    false_safe = predicted_safe & (actual > margin)
    return {
        "predicted_safe_count": int(predicted_safe.sum()),
        "false_safe_count": int(false_safe.sum()),
        "false_safe_rate": float(false_safe.sum() / max(1, predicted_safe.sum())),
    }

