from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from concordia.feasibility.benefit import BenefitModel
from concordia.feasibility.calibration_v4 import ProbabilityCalibrator
from concordia.feasibility.calibration_v5 import RegimeProbabilityCalibrator
from concordia.feasibility.micro_correction import MicroscopicCorrectionModel
from concordia.feasibility.models_v5 import V5SuccessModel
from concordia.safety.microscopic_veto import MicroscopicSafetyVeto


@dataclass(frozen=True)
class V5PredictionBatch:
    success_probability: np.ndarray
    analytical_benefit: np.ndarray
    corrected_microscopic_benefit: np.ndarray
    microscopic_success_probability: np.ndarray
    microscopic_safety_probability_mean: np.ndarray
    microscopic_safety_probability_upper: np.ndarray


@dataclass
class V5ModelBundle:
    success_model: V5SuccessModel
    success_calibrator: RegimeProbabilityCalibrator
    benefit_model: BenefitModel
    micro_correction: MicroscopicCorrectionModel
    micro_calibrator: ProbabilityCalibrator
    micro_safety_veto: MicroscopicSafetyVeto

    @classmethod
    def from_packages(
        cls,
        analytical: Mapping[str, Any],
        micro_correction: Mapping[str, Any],
        micro_safety: Mapping[str, Any],
    ) -> "V5ModelBundle":
        return cls(
            V5SuccessModel.from_dict(analytical["model"]),
            RegimeProbabilityCalibrator.from_dict(analytical["calibrator"]),
            BenefitModel.from_dict(analytical["benefit_model"]),
            MicroscopicCorrectionModel.from_dict(micro_correction["model"]),
            ProbabilityCalibrator.from_dict(micro_correction["calibrator"]),
            MicroscopicSafetyVeto.from_dict(micro_safety["veto"]),
        )

    def predict(
        self, matrix: np.ndarray, regimes: Sequence[str]
    ) -> V5PredictionBatch:
        analytical_matrix = np.asarray(matrix, dtype=float)
        raw = self.success_model.predict_proba(analytical_matrix, regimes)
        probability = self.success_calibrator.predict(raw, regimes)
        benefit = self.benefit_model.predict(analytical_matrix)
        micro_matrix = np.column_stack((analytical_matrix, probability, benefit))
        corrected, micro_raw = self.micro_correction.predict(micro_matrix, benefit)
        micro_probability = self.micro_calibrator.predict(micro_raw)
        safety_mean, safety_upper = self.micro_safety_veto.predict(micro_matrix)
        return V5PredictionBatch(
            probability,
            benefit,
            corrected,
            micro_probability,
            safety_mean,
            safety_upper,
        )
