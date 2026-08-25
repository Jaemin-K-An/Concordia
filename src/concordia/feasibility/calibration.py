from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def _auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positive = int(labels.sum())
    negative = len(labels) - positive
    if not positive or not negative:
        return None
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels == 1].sum() - positive * (positive + 1) / 2) / (positive * negative))


def _pr_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if labels.sum() == 0:
        return None
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    precision = np.cumsum(sorted_labels) / np.arange(1, len(labels) + 1)
    recall_step = sorted_labels / labels.sum()
    return float(np.sum(precision * recall_step))


def classification_metrics(
    labels: Sequence[int], probabilities: Sequence[float], threshold: float = 0.5
) -> dict:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predicted = probabilities >= threshold
    tp = int(np.sum((labels == 1) & predicted))
    fp = int(np.sum((labels == 0) & predicted))
    tn = int(np.sum((labels == 0) & ~predicted))
    fn = int(np.sum((labels == 1) & ~predicted))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    calibration = []
    errors = []
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        members = (probabilities >= lower) & (
            probabilities <= upper if upper >= 1.0 else probabilities < upper
        )
        if members.any():
            predicted_mean = float(probabilities[members].mean())
            observed = float(labels[members].mean())
            count = int(members.sum())
            errors.append(abs(predicted_mean - observed) * count / len(labels))
            calibration.append(
                {
                    "mean_predicted_probability": predicted_mean,
                    "observed_win_rate": observed,
                    "count": count,
                }
            )
    return {
        "roc_auc": _auc(labels, probabilities),
        "pr_auc": _pr_auc(labels, probabilities),
        "brier_score": float(np.mean((probabilities - labels) ** 2)),
        "ece": float(sum(errors)),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "calibration_curve": calibration,
    }


def select_intervention_threshold(
    labels: Sequence[int],
    probabilities: Sequence[float],
    uncertainties: Sequence[float],
    *,
    precision_target: float,
    coverage_target: float,
    maximum_uncertainty: float,
    candidates: Sequence[float],
) -> dict:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    uncertainties = np.asarray(uncertainties, dtype=float)
    rows = []
    for threshold in candidates:
        intervene = (probabilities >= threshold) & (uncertainties <= maximum_uncertainty)
        count = int(intervene.sum())
        precision = float(labels[intervene].mean()) if count else 0.0
        coverage = count / len(labels)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": precision,
                "coverage": coverage,
                "intervention_count": count,
                "meets_precision": precision >= precision_target,
                "meets_coverage": coverage >= coverage_target,
            }
        )
    eligible = [row for row in rows if row["meets_precision"]]
    selected = max(
        eligible or rows,
        key=lambda row: (
            row["coverage"] if eligible else row["precision"],
            row["precision"],
            -row["threshold"],
        ),
    )
    return {"selected": selected, "risk_coverage_curve": rows}


def wilson_interval(successes: int, count: int, confidence: float = 0.95) -> tuple[float, float]:
    if count == 0:
        return (0.0, 1.0)
    z = 1.959963984540054 if math.isclose(confidence, 0.95) else 1.96
    probability = successes / count
    denominator = 1.0 + z**2 / count
    center = (probability + z**2 / (2 * count)) / denominator
    radius = z * math.sqrt(
        probability * (1.0 - probability) / count + z**2 / (4 * count**2)
    ) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)
