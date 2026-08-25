from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from concordia.feasibility.benefit import BenefitModel
from concordia.feasibility.models import FeasibilityModel


@dataclass
class SafetyPredictionModel:
    delta_upper_model: BenefitModel
    violation_model: FeasibilityModel | None
    constant_violation_probability: float
    safety_delta: float

    @classmethod
    def fit(
        cls,
        matrix: np.ndarray,
        safety_difference: np.ndarray,
        feature_names: Sequence[str],
        safety_delta: float,
        seed: int,
    ) -> "SafetyPredictionModel":
        values = np.asarray(safety_difference, dtype=float)
        labels = (values > safety_delta).astype(int)
        upper = BenefitModel(
            "safety_q90_boosting", "boosting", quantile=0.90, iterations=140
        ).fit(matrix, values)
        classifier = None
        if len(np.unique(labels)) == 2:
            classifier = FeasibilityModel(
                "safety_violation_logistic",
                "logistic",
                tuple(feature_names),
                regularization=0.03,
                seed=seed,
            ).fit(matrix, labels)
        return cls(upper, classifier, float(labels.mean()), safety_delta)

    def predict(self, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        upper = self.delta_upper_model.predict(matrix)
        probability = (
            self.violation_model.predict_proba(matrix)
            if self.violation_model is not None
            else np.full(len(matrix), self.constant_violation_probability)
        )
        return upper, probability

    def to_dict(self) -> dict[str, Any]:
        return {
            "delta_upper_model": self.delta_upper_model.to_dict(),
            "violation_model": self.violation_model.to_dict() if self.violation_model else None,
            "constant_violation_probability": self.constant_violation_probability,
            "safety_delta": self.safety_delta,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SafetyPredictionModel":
        classifier = value.get("violation_model")
        return cls(
            BenefitModel.from_dict(value["delta_upper_model"]),
            FeasibilityModel.from_dict(classifier) if classifier else None,
            float(value["constant_violation_probability"]),
            float(value["safety_delta"]),
        )


def false_safe_rate(
    actual_difference: Sequence[float],
    predicted_upper: Sequence[float],
    safety_delta: float,
) -> dict:
    actual = np.asarray(actual_difference, dtype=float)
    predicted = np.asarray(predicted_upper, dtype=float)
    predicted_safe = predicted <= safety_delta
    actually_unsafe = actual > safety_delta
    count = int(np.sum(predicted_safe & actually_unsafe))
    return {
        "false_safe_count": count,
        "false_safe_rate": count / max(1, len(actual)),
        "predicted_safe_count": int(predicted_safe.sum()),
        "unsafe_count": int(actually_unsafe.sum()),
    }
