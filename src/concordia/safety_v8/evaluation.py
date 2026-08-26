from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def pr_curve(labels: Sequence[int], probabilities: Sequence[float]) -> list[dict]:
    y = np.asarray(labels, dtype=bool)
    p = np.asarray(probabilities, dtype=float)
    thresholds = np.unique(np.concatenate(([0.0], p, [1.0])))[::-1]
    output = []
    for threshold in thresholds:
        predicted = p >= threshold
        tp = int((predicted & y).sum())
        fp = int((predicted & ~y).sum())
        output.append(
            {
                "threshold": float(threshold),
                "precision": _safe_div(tp, tp + fp),
                "recall": _safe_div(tp, int(y.sum())),
            }
        )
    return output


def average_precision(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    y = np.asarray(labels, dtype=bool)
    p = np.asarray(probabilities, dtype=float)
    if not y.any():
        return 0.0
    order = np.argsort(-p, kind="stable")
    ranked = y[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked].sum() / y.sum())


def classification_metrics(
    labels: Sequence[int], probabilities: Sequence[float], threshold: float
) -> dict:
    y = np.asarray(labels, dtype=bool)
    p = np.asarray(probabilities, dtype=float)
    predicted = p >= threshold
    tp = int((predicted & y).sum())
    fp = int((predicted & ~y).sum())
    fn = int((~predicted & y).sum())
    tn = int((~predicted & ~y).sum())
    predicted_safe = ~predicted
    false_safe = int((predicted_safe & y).sum())
    return {
        "sample_count": len(y),
        "unsafe_count": int(y.sum()),
        "threshold": float(threshold),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "unsafe_recall": _safe_div(tp, tp + fn),
        "unsafe_precision": _safe_div(tp, tp + fp),
        "specificity": _safe_div(tn, tn + fp),
        "false_safe_rate_all_candidates": _safe_div(false_safe, len(y)),
        "false_safe_rate_given_predicted_safe": _safe_div(false_safe, int(predicted_safe.sum())),
        "pr_auc_average_precision": average_precision(y, p),
    }


def critical_group_recall(
    rows: Sequence[Mapping[str, object]], probabilities: Sequence[float], threshold: float
) -> dict:
    p = np.asarray(probabilities, dtype=float)
    output = {}
    for field in ("topology", "penetration"):
        values = [
            str(row["condition"].get(field))
            if field == "topology"
            else f"{float(row['condition'].get(field, 0.0)):.2f}"
            for row in rows
        ]
        for value in sorted(set(values)):
            mask = np.asarray([item == value for item in values])
            y = np.asarray([float(row["outcomes"]["tau_s"]) > 0.25 for row in rows])
            unsafe = mask & y
            key = f"{field}={value}"
            output[key] = {
                "sample_count": int(mask.sum()),
                "unsafe_count": int(unsafe.sum()),
                "unsafe_recall": float((unsafe & (p >= threshold)).sum() / unsafe.sum())
                if unsafe.any()
                else None,
            }
    finite = [value["unsafe_recall"] for value in output.values() if value["unsafe_recall"] is not None]
    return {"groups": output, "worst_critical_group_recall": min(finite) if finite else 0.0}
