from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from concordia.feasibility.benefit import BenefitModel
from concordia.feasibility.models import FeasibilityModel


MICRO_ADDITIONAL_FEATURES = (
    "analytical_success_probability",
    "analytical_benefit",
)


@dataclass
class MicroscopicCorrectionModel:
    correction_model: BenefitModel
    success_model: FeasibilityModel
    feature_names: tuple[str, ...]

    @classmethod
    def fit(
        cls,
        matrix: np.ndarray,
        analytical_benefit: Sequence[float],
        microscopic_benefit: Sequence[float],
        microscopic_success: Sequence[int],
        feature_names: Sequence[str],
    ) -> "MicroscopicCorrectionModel":
        analytical = np.asarray(analytical_benefit, dtype=float)
        realized = np.asarray(microscopic_benefit, dtype=float)
        correction = BenefitModel("micro_delta_ridge", "ridge", penalty=0.05).fit(
            matrix, realized - analytical
        )
        success = FeasibilityModel(
            "micro_success_logistic",
            "logistic",
            tuple(feature_names),
            regularization=0.08,
            iterations=2200,
        ).fit(matrix, np.asarray(microscopic_success, dtype=int))
        return cls(correction, success, tuple(feature_names))

    def predict(
        self, matrix: np.ndarray, analytical_benefit: Sequence[float]
    ) -> tuple[np.ndarray, np.ndarray]:
        corrected = np.asarray(analytical_benefit, dtype=float) + self.correction_model.predict(
            matrix
        )
        return corrected, self.success_model.predict_proba(matrix)

    def to_dict(self) -> dict[str, Any]:
        return {
            "correction_model": self.correction_model.to_dict(),
            "success_model": self.success_model.to_dict(),
            "feature_names": list(self.feature_names),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MicroscopicCorrectionModel":
        return cls(
            BenefitModel.from_dict(value["correction_model"]),
            FeasibilityModel.from_dict(value["success_model"]),
            tuple(value["feature_names"]),
        )
