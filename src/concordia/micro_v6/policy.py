from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Mapping, Sequence

import numpy as np

from concordia.feasibility.calibration_v4 import ProbabilityCalibrator
from .features import MICRO_V6_FEATURE_SCHEMA
from .modeling import MicroSuccessPredictor, feature_matrix, load_tabular_model, row_regimes


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 1.0
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    rate = successes / total
    denominator = 1.0 + z * z / total
    centre = (rate + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(rate * (1.0 - rate) / total + z * z / (4 * total * total))
    radius /= denominator
    return max(0.0, centre - radius), min(1.0, centre + radius)


def selective_metrics(
    rows: Sequence[Mapping[str, object]], selected: Sequence[bool]
) -> dict[str, Any]:
    chosen = np.asarray(selected, dtype=bool)
    labels = np.asarray([bool(row["label"]["safe_micro_success"]) for row in rows])
    unsafe = np.asarray([not bool(row["label"]["safety_pass"]) for row in rows])
    count = int(chosen.sum())
    successes = int((chosen & labels).sum())
    opportunities = int(labels.sum())
    lower, upper = wilson_interval(successes, count)
    return {
        "sample_count": len(rows),
        "intervention_count": count,
        "success_count": successes,
        "failure_count": count - successes,
        "precision": successes / count if count else 0.0,
        "coverage": count / len(rows) if rows else 0.0,
        "opportunity_count": opportunities,
        "opportunity_recovery_rate": successes / opportunities if opportunities else 0.0,
        "missed_opportunity_rate": 1.0 - successes / opportunities if opportunities else 0.0,
        "safety_violation_count": int((chosen & unsafe).sum()),
        "false_safe_rate": float((chosen & unsafe).sum() / count) if count else 0.0,
        "precision_wilson_95_lower": lower,
        "precision_wilson_95_upper": upper,
    }


def claim_allowed(
    metrics: Mapping[str, object],
    *,
    minimum_interventions: int,
    required_precision: float,
) -> bool:
    return bool(
        int(metrics["intervention_count"]) >= minimum_interventions
        and float(metrics["precision"]) >= required_precision
        and int(metrics["safety_violation_count"]) == 0
    )


@dataclass
class V6Policy:
    composite: MicroSuccessPredictor
    composite_calibration: ProbabilityCalibrator
    benefit_model: Any
    benefit_calibration: ProbabilityCalibrator
    safety_model: Any
    safety_calibration: ProbabilityCalibrator
    architecture: str
    success_threshold: float
    safety_threshold: float
    stage1_threshold: float
    conformal: bool = False

    def probabilities(
        self, rows: Sequence[Mapping[str, object]]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        matrix = feature_matrix(rows, self.composite.feature_names)
        regimes = row_regimes(rows)
        composite = self.composite_calibration.predict(
            self.composite.predict_proba(matrix, regimes)
        )
        benefit = self.benefit_calibration.predict(self.benefit_model.predict_proba(matrix))
        unsafe = self.safety_calibration.predict(self.safety_model.predict_proba(matrix))
        return composite, benefit, unsafe

    def decide(self, rows: Sequence[Mapping[str, object]]) -> list[dict[str, Any]]:
        composite, benefit, unsafe = self.probabilities(rows)
        decisions = []
        for index, row in enumerate(rows):
            analytical = float(row["features_pre_decision"]["analytical_success_probability"])
            stage1_pass = analytical >= self.stage1_threshold
            score = benefit[index] if self.architecture == "B_benefit_plus_safety_veto" else composite[index]
            safety_veto = (
                self.architecture in {
                    "B_benefit_plus_safety_veto",
                    "C_composite_plus_safety_veto",
                }
                and unsafe[index] > self.safety_threshold
            )
            intervene = bool(stage1_pass and score >= self.success_threshold and not safety_veto)
            reason = "intervene"
            if not stage1_pass:
                reason = "analytical_screen"
            elif score < self.success_threshold:
                reason = "micro_abstain"
            elif safety_veto:
                reason = "safety_veto"
            decisions.append(
                {
                    "case_id": row["case_id"],
                    "intervene": intervene,
                    "executed_policy": "B6" if intervene else "B1",
                    "reason": reason,
                    "composite_probability": float(composite[index]),
                    "benefit_probability": float(benefit[index]),
                    "unsafe_probability": float(unsafe[index]),
                    "analytical_probability": analytical,
                }
            )
        return decisions

    def to_dict(self) -> dict[str, Any]:
        return {
            "composite": self.composite.to_dict(),
            "composite_calibration": self.composite_calibration.to_dict(),
            "benefit_model": self.benefit_model.to_dict(),
            "benefit_calibration": self.benefit_calibration.to_dict(),
            "safety_model": self.safety_model.to_dict(),
            "safety_calibration": self.safety_calibration.to_dict(),
            "architecture": self.architecture,
            "success_threshold": self.success_threshold,
            "safety_threshold": self.safety_threshold,
            "stage1_threshold": self.stage1_threshold,
            "conformal": self.conformal,
            "feature_schema": list(MICRO_V6_FEATURE_SCHEMA),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "V6Policy":
        return cls(
            MicroSuccessPredictor.from_dict(value["composite"]),
            ProbabilityCalibrator.from_dict(value["composite_calibration"]),
            load_tabular_model(value["benefit_model"]),
            ProbabilityCalibrator.from_dict(value["benefit_calibration"]),
            load_tabular_model(value["safety_model"]),
            ProbabilityCalibrator.from_dict(value["safety_calibration"]),
            str(value["architecture"]),
            float(value["success_threshold"]),
            float(value["safety_threshold"]),
            float(value["stage1_threshold"]),
            bool(value.get("conformal", False)),
        )


def selected_mask(decisions: Sequence[Mapping[str, object]]) -> np.ndarray:
    return np.asarray([bool(value["intervene"]) for value in decisions], dtype=bool)
