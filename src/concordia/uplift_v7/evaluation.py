from __future__ import annotations

import math
from statistics import NormalDist
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 1.0
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    rate = successes / total
    denominator = 1.0 + z * z / total
    centre = (rate + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total))
    radius /= denominator
    return max(0.0, centre - radius), min(1.0, centre + radius)


def regression_metrics(actual: Sequence[float], predicted: Sequence[float]) -> dict:
    y = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    correlation = spearmanr(y, p).statistic if len(y) > 1 else 0.0
    correlation = 0.0 if not np.isfinite(correlation) else float(correlation)
    positive = y >= 0.01
    return {
        "mae": float(np.mean(np.abs(y - p))),
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "sign_accuracy": float(np.mean(np.sign(y) == np.sign(p))),
        "positive_uplift_recall": float(np.mean(p[positive] >= 0.01)) if positive.any() else 0.0,
        "spearman": correlation,
    }


def deployment_metrics(
    rows: Sequence[Mapping[str, object]], selected: Sequence[bool]
) -> dict:
    mask = np.asarray(selected, dtype=bool)
    success = np.asarray([bool(row["outcomes"]["safe_micro_success"]) for row in rows])
    traffic = np.asarray([float(row["outcomes"]["tau_t_relative"]) for row in rows])
    unsafe = np.asarray([float(row["outcomes"]["tau_s"]) > 0.25 for row in rows])
    regret = np.asarray([float(row["outcomes"]["max_regret"]) > 0.08 for row in rows])
    count = int(mask.sum())
    successes = int((mask & success).sum())
    opportunities = int(success.sum())
    lower, upper = wilson_interval(successes, count)
    return {
        "sample_count": len(rows),
        "intervention_count": count,
        "success_count": successes,
        "failure_count": count - successes,
        "deployment_precision": successes / count if count else 0.0,
        "uplift_precision": float((mask & (traffic >= 0.01)).sum() / count) if count else 0.0,
        "coverage": count / len(rows) if rows else 0.0,
        "opportunity_count": opportunities,
        "opportunity_recovery_rate": successes / opportunities if opportunities else 0.0,
        "safety_violation_count": int((mask & unsafe).sum()),
        "regret_violation_count": int((mask & regret).sum()),
        "false_safe_rate": float((mask & unsafe).sum() / count) if count else 0.0,
        "precision_wilson_95_lower": lower,
        "precision_wilson_95_upper": upper,
        "mean_selected_relative_uplift": float(traffic[mask].mean()) if count else 0.0,
    }


def effect_calibration(actual: Sequence[float], predicted: Sequence[float]) -> list[dict]:
    y = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    bins = (
        ("below_0", -np.inf, 0.0),
        ("0_to_1_percent", 0.0, 0.01),
        ("1_to_3_percent", 0.01, 0.03),
        ("above_3_percent", 0.03, np.inf),
    )
    output = []
    for name, lower, upper in bins:
        mask = (p >= lower) & (p < upper)
        output.append(
            {
                "bin": name,
                "count": int(mask.sum()),
                "mean_predicted_uplift": float(p[mask].mean()) if mask.any() else None,
                "mean_realized_uplift": float(y[mask].mean()) if mask.any() else None,
            }
        )
    return output


def cumulative_gain(actual: Sequence[float], predicted: Sequence[float]) -> list[dict]:
    y = np.asarray(actual, dtype=float)
    order = np.argsort(-np.asarray(predicted, dtype=float))
    output = []
    for fraction in (0.05, 0.10, 0.20, 0.30, 0.50, 1.0):
        count = max(1, int(math.ceil(fraction * len(y))))
        selected = y[order[:count]]
        output.append(
            {
                "fraction": fraction,
                "count": count,
                "mean_realized_uplift": float(selected.mean()),
                "cumulative_realized_uplift": float(selected.sum()),
            }
        )
    return output

