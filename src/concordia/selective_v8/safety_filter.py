from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from concordia.feasibility.calibration_v4 import ProbabilityCalibrator
from concordia.safety_v8.classifier import SafetyClassifier
from concordia.safety_v8.features import feature_matrix


@dataclass
class CalibratedSafetyFilter:
    classifier: SafetyClassifier
    calibrator: ProbabilityCalibrator
    unsafe_probability_threshold: float

    def probabilities(self, rows: Sequence[Mapping[str, object]]) -> np.ndarray:
        raw = self.classifier.predict_proba(feature_matrix(rows, self.classifier.feature_names))
        return self.calibrator.predict(raw)

    def safe(self, rows: Sequence[Mapping[str, object]]) -> np.ndarray:
        return self.probabilities(rows) <= self.unsafe_probability_threshold

    def to_dict(self) -> dict[str, Any]:
        return {
            "classifier": self.classifier.to_dict(),
            "calibrator": self.calibrator.to_dict(),
            "unsafe_probability_threshold": self.unsafe_probability_threshold,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CalibratedSafetyFilter":
        return cls(
            SafetyClassifier.from_dict(value["classifier"]),
            ProbabilityCalibrator.from_dict(value["calibrator"]),
            float(value["unsafe_probability_threshold"]),
        )

