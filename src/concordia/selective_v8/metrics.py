from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr

from concordia.uplift_v7.evaluation import deployment_metrics


def ranking_metrics(rows: Sequence[Mapping[str, object]], scores: Sequence[float]) -> dict:
    actual = np.asarray([float(row["outcomes"]["tau_t_relative"]) for row in rows])
    score = np.asarray(scores, dtype=float)
    order = np.argsort(-score, kind="stable")
    correlation = spearmanr(actual, score).statistic if len(actual) > 1 else 0.0
    correlation = float(correlation) if np.isfinite(correlation) else 0.0
    population = float(actual.mean())
    top_k = {}
    cumulative = []
    for fraction in (0.05, 0.10, 0.15, 0.20, 0.25):
        count = max(1, int(math.ceil(fraction * len(rows))))
        selected = actual[order[:count]]
        top_k[f"top_{int(100 * fraction)}_percent"] = {
            "count": count,
            "mean_realized_uplift": float(selected.mean()),
            "uplift_over_population_mean": float(selected.mean() - population),
            "cumulative_uplift": float(selected.sum()),
        }
        cumulative.append({"fraction": fraction, "count": count, "gain": float(selected.sum())})
    positive = np.maximum(actual, 0.0)
    ideal = np.sort(positive)[::-1]
    discounts = 1.0 / np.log2(np.arange(2, len(actual) + 2))
    dcg = float((positive[order] * discounts).sum())
    idcg = float((ideal * discounts).sum())
    return {
        "sample_count": len(rows),
        "population_mean_uplift": population,
        "spearman": correlation,
        "top_k": top_k,
        "ndcg": dcg / idcg if idcg else 0.0,
        "cumulative_gain": cumulative,
    }


def filtered_policy_metrics(
    rows: Sequence[Mapping[str, object]], selected: Sequence[bool], traffic_screen: Sequence[bool]
) -> dict:
    metrics = deployment_metrics(rows, selected)
    selected_mask = np.asarray(selected, dtype=bool)
    screen = np.asarray(traffic_screen, dtype=bool)
    unsafe = np.asarray([float(row["outcomes"]["tau_s"]) > 0.25 for row in rows])
    success = np.asarray([bool(row["outcomes"]["safe_micro_success"]) for row in rows])
    removed_unsafe = int((screen & unsafe & ~selected_mask).sum())
    screened_unsafe = int((screen & unsafe).sum())
    retained_success = int((screen & success & selected_mask).sum())
    screened_success = int((screen & success).sum())
    metrics.update(
        {
            "opportunity_realization_rate": metrics["opportunity_recovery_rate"],
            "safety_filter_effectiveness": removed_unsafe / screened_unsafe if screened_unsafe else 1.0,
            "safe_success_retention": retained_success / screened_success if screened_success else 0.0,
            "screened_unsafe_count": screened_unsafe,
            "screened_safe_success_count": screened_success,
        }
    )
    return metrics
