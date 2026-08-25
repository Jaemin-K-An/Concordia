from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from concordia.feasibility.benefit import BenefitModel
from concordia.feasibility.calibration_v4 import ProbabilityCalibrator
from concordia.feasibility.models_v4 import V4BootstrapEnsemble
from concordia.feasibility.safety_prediction import SafetyPredictionModel
from concordia.selective.score import expected_safe_intervention_value, risk_adjusted_esiv


@dataclass
class V4PredictionBundle:
    ensemble: V4BootstrapEnsemble
    calibrator: ProbabilityCalibrator
    benefit_mean: BenefitModel
    benefit_lower: BenefitModel
    safety_model: SafetyPredictionModel
    success_lower_z: float
    safety_upper_adjustment: float
    safety_probability_buffer: float

    @classmethod
    def from_packages(
        cls,
        probability: Mapping[str, Any],
        benefit: Mapping[str, Any],
        safety: Mapping[str, Any],
    ) -> "V4PredictionBundle":
        return cls(
            V4BootstrapEnsemble.from_dict(probability["ensemble"]),
            ProbabilityCalibrator.from_dict(probability["calibrator"]),
            BenefitModel.from_dict(benefit["mean_model"]),
            BenefitModel.from_dict(benefit["lower_model"]),
            SafetyPredictionModel.from_dict(safety["model"]),
            float(probability["uncertainty_contract"]["success_probability_lower_z"]),
            float(safety["upper_adjustment"]),
            float(safety["probability_upper_buffer"]),
        )

    def predict(self, matrix: np.ndarray) -> dict[str, np.ndarray]:
        raw, uncertainty, raw_quantile = self.ensemble.predict(matrix)
        probability = self.calibrator.predict(raw)
        probability_quantile = self.calibrator.predict(raw_quantile)
        probability_lower = np.clip(
            np.minimum(
                probability_quantile,
                probability - self.success_lower_z * uncertainty,
            ),
            0.0,
            1.0,
        )
        benefit = self.benefit_mean.predict(matrix)
        benefit_lower = self.benefit_lower.predict(matrix)
        safety_upper, safety_probability = self.safety_model.predict(matrix)
        safety_upper = safety_upper + self.safety_upper_adjustment
        safety_probability_upper = np.clip(
            safety_probability + self.safety_probability_buffer, 0.0, 1.0
        )
        esiv = np.asarray(
            [
                expected_safe_intervention_value(p, b, r)
                for p, b, r in zip(probability, benefit, safety_probability)
            ]
        )
        esiv_lower = np.asarray(
            [
                risk_adjusted_esiv(p, b, r)
                for p, b, r in zip(
                    probability_lower, benefit_lower, safety_probability_upper
                )
            ]
        )
        return {
            "raw_probability": raw,
            "probability": probability,
            "probability_lower": probability_lower,
            "uncertainty": uncertainty,
            "expected_benefit": benefit,
            "benefit_lower": benefit_lower,
            "safety_difference_upper": safety_upper,
            "safety_failure_probability": safety_probability,
            "safety_failure_probability_upper": safety_probability_upper,
            "esiv": esiv,
            "esiv_lower": esiv_lower,
        }
