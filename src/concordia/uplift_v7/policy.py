from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .conformal import ConformalAdjustments
from .paired_dataset import UPLIFT_V7_FEATURE_SCHEMA, feature_matrix
from .quantiles import BootstrapCausalEnsemble, BootstrapRegressionEnsemble
from .treatment_effect import CausalEffectLearner
from .learners import RegressionModel


@dataclass
class UpliftPolicy:
    traffic_model: CausalEffectLearner
    safety_model: RegressionModel
    regret_model: RegressionModel
    traffic_bootstrap: BootstrapCausalEnsemble
    safety_bootstrap: BootstrapRegressionEnsemble
    regret_bootstrap: BootstrapRegressionEnsemble
    conformal: ConformalAdjustments
    interval_method: str
    traffic_lcb_threshold: float
    safety_ucb_threshold: float
    regret_ucb_threshold: float

    def predict_bounds(self, rows: Sequence[Mapping[str, object]]) -> dict[str, np.ndarray]:
        matrix = feature_matrix(rows, self.traffic_model.feature_names)
        traffic_mean = self.traffic_model.predict(matrix)
        safety_mean = self.safety_model.predict(matrix)
        regret_mean = self.regret_model.predict(matrix)
        if self.interval_method == "bootstrap_quantile":
            _, traffic_lower, traffic_upper = self.traffic_bootstrap.interval(matrix, 0.10, 0.90)
            _, safety_lower, safety_upper = self.safety_bootstrap.interval(matrix, 0.10, 0.90)
            _, regret_lower, regret_upper = self.regret_bootstrap.interval(matrix, 0.10, 0.90)
        else:
            traffic_lower = traffic_mean - self.conformal.traffic_radius
            traffic_upper = traffic_mean + self.conformal.traffic_radius
            safety_lower = safety_mean - self.conformal.safety_upper_adjustment
            safety_upper = safety_mean + self.conformal.safety_upper_adjustment
            regret_lower = regret_mean - self.conformal.regret_upper_adjustment
            regret_upper = regret_mean + self.conformal.regret_upper_adjustment
        return {
            "traffic_mean": traffic_mean,
            "traffic_lower": traffic_lower,
            "traffic_upper": traffic_upper,
            "safety_mean": safety_mean,
            "safety_lower": safety_lower,
            "safety_upper": safety_upper,
            "regret_mean": regret_mean,
            "regret_lower": regret_lower,
            "regret_upper": regret_upper,
        }

    def decide(self, rows: Sequence[Mapping[str, object]]) -> list[dict[str, Any]]:
        bounds = self.predict_bounds(rows)
        output = []
        for index, row in enumerate(rows):
            traffic_pass = bounds["traffic_lower"][index] > self.traffic_lcb_threshold
            safety_pass = bounds["safety_upper"][index] <= self.safety_ucb_threshold
            regret_pass = bounds["regret_upper"][index] <= self.regret_ucb_threshold
            legal = bool(row.get("legal_executable_predecision", True))
            intervene = bool(traffic_pass and safety_pass and regret_pass and legal)
            reason = "intervene"
            if not traffic_pass:
                reason = "traffic_uplift_uncertain"
            elif not safety_pass:
                reason = "safety_effect_veto"
            elif not regret_pass:
                reason = "regret_veto"
            elif not legal:
                reason = "illegal_or_unexecutable"
            output.append(
                {
                    "pair_id": row["pair_id"],
                    "intervene": intervene,
                    "executed_policy": "Adaptive" if intervene else "B1",
                    "reason": reason,
                    **{name: float(values[index]) for name, values in bounds.items()},
                }
            )
        return output

    def to_dict(self) -> dict[str, Any]:
        return {
            "traffic_model": self.traffic_model.to_dict(),
            "safety_model": self.safety_model.to_dict(),
            "regret_model": self.regret_model.to_dict(),
            "traffic_bootstrap": self.traffic_bootstrap.to_dict(),
            "safety_bootstrap": self.safety_bootstrap.to_dict(),
            "regret_bootstrap": self.regret_bootstrap.to_dict(),
            "conformal": self.conformal.to_dict(),
            "interval_method": self.interval_method,
            "traffic_lcb_threshold": self.traffic_lcb_threshold,
            "safety_ucb_threshold": self.safety_ucb_threshold,
            "regret_ucb_threshold": self.regret_ucb_threshold,
            "feature_schema": list(UPLIFT_V7_FEATURE_SCHEMA),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "UpliftPolicy":
        return cls(
            CausalEffectLearner.from_dict(value["traffic_model"]),
            RegressionModel.from_dict(value["safety_model"]),
            RegressionModel.from_dict(value["regret_model"]),
            BootstrapCausalEnsemble.from_dict(value["traffic_bootstrap"]),
            BootstrapRegressionEnsemble.from_dict(value["safety_bootstrap"]),
            BootstrapRegressionEnsemble.from_dict(value["regret_bootstrap"]),
            ConformalAdjustments.from_dict(value["conformal"]),
            str(value["interval_method"]),
            float(value["traffic_lcb_threshold"]),
            float(value["safety_ucb_threshold"]),
            float(value["regret_ucb_threshold"]),
        )

