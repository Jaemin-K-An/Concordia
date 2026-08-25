from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from concordia.errors import ValidationError


PHANTOM_FEATURES = (
    "density",
    "mean_speed",
    "speed_cv",
    "acceleration_variance",
    "headway_variance",
    "flow",
    "saturation",
    "geometry_complexity",
)


@dataclass(frozen=True)
class CalibrationMetrics:
    roc_auc: float
    pr_auc: float
    brier_score: float
    expected_calibration_error: float
    recall: float
    false_negative_rate: float
    threshold: float
    sample_count: int


def _validate_xy(features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=int)
    if x.ndim != 2 or x.shape[1] != len(PHANTOM_FEATURES) or y.shape != (len(x),):
        raise ValidationError("phantom predictor feature/label dimensions are invalid")
    if len(x) < 4 or not np.all(np.isfinite(x)) or not set(np.unique(y)).issubset({0, 1}):
        raise ValidationError("phantom predictor data must be finite binary labeled samples")
    if len(np.unique(y)) < 2:
        raise ValidationError("phantom predictor calibration requires both outcome classes")
    return x, y


def calibration_metrics(
    labels: Sequence[int], probabilities: Sequence[float], threshold: float = 0.5
) -> CalibrationMetrics:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if y.shape != p.shape or len(y) < 2 or not 0 < threshold < 1:
        raise ValidationError("calibration metric inputs are invalid")
    if np.any((p < 0) | (p > 1)) or len(np.unique(y)) < 2:
        raise ValidationError("calibration probabilities/classes are invalid")
    positive = p[y == 1]
    negative = p[y == 0]
    comparisons = sum((value > negative).sum() + 0.5 * (value == negative).sum() for value in positive)
    roc_auc = float(comparisons / (len(positive) * len(negative)))
    order = np.argsort(-p)
    sorted_y = y[order]
    true_positive = np.cumsum(sorted_y)
    false_positive = np.cumsum(1 - sorted_y)
    recall = true_positive / max(1, int(y.sum()))
    precision = true_positive / np.maximum(1, true_positive + false_positive)
    pr_x = np.r_[0.0, recall]
    pr_y = np.r_[1.0, precision]
    pr_auc = float(np.sum((pr_y[1:] + pr_y[:-1]) * 0.5 * np.diff(pr_x)))
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        members = (p >= lower) & (p <= upper if upper == 1 else p < upper)
        if members.any():
            ece += float(members.mean()) * abs(float(p[members].mean() - y[members].mean()))
    predictions = p >= threshold
    false_negatives = int(((predictions == 0) & (y == 1)).sum())
    return CalibrationMetrics(
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        brier_score=float(np.mean((p - y) ** 2)),
        expected_calibration_error=ece,
        recall=1.0 - false_negatives / max(1, int(y.sum())),
        false_negative_rate=false_negatives / max(1, int(y.sum())),
        threshold=threshold,
        sample_count=len(y),
    )


class LogisticPhantomJamRiskPredictor:
    def __init__(self, learning_rate: float = 0.05, iterations: int = 3000, l2: float = 1e-3) -> None:
        if learning_rate <= 0 or iterations < 1 or l2 < 0:
            raise ValidationError("logistic predictor hyperparameters are invalid")
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.l2 = l2
        self.mean: Optional[np.ndarray] = None
        self.scale: Optional[np.ndarray] = None
        self.coefficients: Optional[np.ndarray] = None
        self.intercept = 0.0

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "LogisticPhantomJamRiskPredictor":
        x, y = _validate_xy(features, labels)
        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0)
        self.scale[self.scale < 1e-12] = 1.0
        normalized = (x - self.mean) / self.scale
        coefficients = np.zeros(x.shape[1], dtype=float)
        intercept = math.log((float(y.mean()) + 1e-6) / (1.0 - float(y.mean()) + 1e-6))
        for _ in range(self.iterations):
            logits = np.clip(normalized @ coefficients + intercept, -30, 30)
            probability = 1.0 / (1.0 + np.exp(-logits))
            error = probability - y
            coefficients -= self.learning_rate * (
                normalized.T @ error / len(x) + self.l2 * coefficients
            )
            intercept -= self.learning_rate * float(error.mean())
        self.coefficients = coefficients
        self.intercept = intercept
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.coefficients is None or self.mean is None or self.scale is None:
            raise ValidationError("logistic phantom predictor is not fitted")
        x = np.asarray(features, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[1] != len(PHANTOM_FEATURES):
            raise ValidationError("phantom prediction feature dimension is invalid")
        logits = np.clip(((x - self.mean) / self.scale) @ self.coefficients + self.intercept, -30, 30)
        return 1.0 / (1.0 + np.exp(-logits))

    def model_card(self) -> Dict[str, object]:
        if self.coefficients is None or self.mean is None or self.scale is None:
            raise ValidationError("cannot serialize an unfitted predictor")
        return {
            "model": "logistic_regression",
            "features": list(PHANTOM_FEATURES),
            "coefficients": self.coefficients.tolist(),
            "intercept": self.intercept,
            "normalization_mean": self.mean.tolist(),
            "normalization_scale": self.scale.tolist(),
        }


@dataclass(frozen=True)
class DecisionStump:
    feature_index: int
    threshold: float
    lower_probability: float
    upper_probability: float


class StumpEnsemblePhantomJamRiskPredictor:
    """Interpretable tree-based comparator built from calibrated decision stumps."""

    def __init__(self, maximum_stumps: int = 12) -> None:
        if maximum_stumps < 1:
            raise ValidationError("stump ensemble size must be positive")
        self.maximum_stumps = maximum_stumps
        self.stumps: Tuple[DecisionStump, ...] = ()

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "StumpEnsemblePhantomJamRiskPredictor":
        x, y = _validate_xy(features, labels)
        candidates = []
        for feature_index in range(x.shape[1]):
            for threshold in np.unique(np.quantile(x[:, feature_index], np.linspace(0.1, 0.9, 9))):
                lower = y[x[:, feature_index] <= threshold]
                upper = y[x[:, feature_index] > threshold]
                if not len(lower) or not len(upper):
                    continue
                lower_probability = (float(lower.sum()) + 1.0) / (len(lower) + 2.0)
                upper_probability = (float(upper.sum()) + 1.0) / (len(upper) + 2.0)
                prediction = np.where(
                    x[:, feature_index] <= threshold, lower_probability, upper_probability
                )
                brier = float(np.mean((prediction - y) ** 2))
                candidates.append(
                    (
                        brier,
                        DecisionStump(
                            feature_index,
                            float(threshold),
                            lower_probability,
                            upper_probability,
                        ),
                    )
                )
        if not candidates:
            raise ValidationError("no valid decision stump could be fitted")
        self.stumps = tuple(stump for _, stump in sorted(candidates, key=lambda item: item[0])[: self.maximum_stumps])
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if not self.stumps:
            raise ValidationError("stump phantom predictor is not fitted")
        x = np.asarray(features, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        predictions = [
            np.where(
                x[:, stump.feature_index] <= stump.threshold,
                stump.lower_probability,
                stump.upper_probability,
            )
            for stump in self.stumps
        ]
        return np.mean(predictions, axis=0)

    def model_card(self) -> Dict[str, object]:
        return {
            "model": "decision_stump_ensemble",
            "features": list(PHANTOM_FEATURES),
            "stumps": [asdict(stump) for stump in self.stumps],
        }
