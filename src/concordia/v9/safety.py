from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from concordia.feasibility.calibration_v4 import ProbabilityCalibrator
from concordia.safety_v8.classifier import SafetyClassifier

from .action_features import STATE_ACTION_FEATURE_SCHEMA, feature_matrix


def unsafe_label(row: Mapping[str, object]) -> int:
    outcome = row["outcomes"]
    return int(
        float(outcome["tau_s"]) > 0.25
        or float(outcome["max_regret"]) > 0.08
        or not bool(outcome["legal"])
    )


@dataclass
class ActionSafetyModel:
    classifier: SafetyClassifier
    calibrator: ProbabilityCalibrator | None = None

    @classmethod
    def build(cls, model_id: str, kind: str, seed: int, positive_weight: float = 3.0):
        return cls(SafetyClassifier(
            model_id, kind, tuple(STATE_ACTION_FEATURE_SCHEMA), seed, positive_weight
        ))

    def fit(self, rows: Sequence[Mapping[str, object]]) -> "ActionSafetyModel":
        self.classifier.fit(feature_matrix(rows), [unsafe_label(row) for row in rows])
        return self

    def raw_probability(self, rows: Sequence[Mapping[str, object]]) -> np.ndarray:
        return self.classifier.predict_proba(feature_matrix(rows))

    def calibrate(
        self, rows: Sequence[Mapping[str, object]], method: str
    ) -> "ActionSafetyModel":
        self.calibrator = ProbabilityCalibrator(method).fit(
            self.raw_probability(rows), [unsafe_label(row) for row in rows]
        )
        return self

    def predict_proba(self, rows: Sequence[Mapping[str, object]]) -> np.ndarray:
        raw = self.raw_probability(rows)
        return self.calibrator.predict(raw) if self.calibrator else raw

    def to_dict(self) -> dict[str, Any]:
        return {
            "classifier": self.classifier.to_dict(),
            "calibrator": self.calibrator.to_dict() if self.calibrator else None,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ActionSafetyModel":
        return cls(
            SafetyClassifier.from_dict(value["classifier"]),
            ProbabilityCalibrator.from_dict(value["calibrator"])
            if value.get("calibrator") else None,
        )
