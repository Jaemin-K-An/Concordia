from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .learners import RegressionModel, build_regression_model


@dataclass
class CausalEffectLearner:
    formulation: str
    kind: str
    feature_names: tuple[str, ...]
    direct_model: RegressionModel | None = None
    control_model: RegressionModel | None = None
    treatment_model: RegressionModel | None = None
    s_model: RegressionModel | None = None
    residual_model: RegressionModel | None = None

    @classmethod
    def fit(
        cls,
        formulation: str,
        kind: str,
        matrix: np.ndarray,
        tau_relative: np.ndarray,
        ttt_b1: np.ndarray,
        ttt_adaptive: np.ndarray,
        generated: np.ndarray,
        feature_names: Sequence[str],
        *,
        seed: int,
    ) -> "CausalEffectLearner":
        names = tuple(feature_names)
        direct = build_regression_model(kind, names, seed, f"{formulation}:{kind}:direct")
        direct.fit(matrix, tau_relative)
        learner = cls(formulation, kind, names, direct_model=direct)
        if formulation == "C0_direct_paired":
            return learner
        y0 = ttt_b1 / np.maximum(generated, 1.0)
        y1 = ttt_adaptive / np.maximum(generated, 1.0)
        if formulation in {"C1_t_learner", "C3_x_learner", "C4_paired_dr"}:
            learner.control_model = build_regression_model(kind, names, seed + 1, "mu0").fit(
                matrix, y0
            )
            learner.treatment_model = build_regression_model(kind, names, seed + 2, "mu1").fit(
                matrix, y1
            )
        if formulation == "C2_s_learner":
            s_names = (*names, "treatment_indicator")
            s_matrix = np.vstack(
                [
                    np.column_stack([matrix, np.zeros(len(matrix))]),
                    np.column_stack([matrix, np.ones(len(matrix))]),
                ]
            )
            learner.s_model = build_regression_model(kind, s_names, seed + 3, "s_learner").fit(
                s_matrix, np.concatenate([y0, y1])
            )
        if formulation == "C4_paired_dr":
            base_effect = learner._potential_effect(matrix)
            learner.residual_model = build_regression_model(
                kind, names, seed + 4, "paired_dr_residual"
            ).fit(matrix, tau_relative - base_effect)
        return learner

    def _potential_effect(self, matrix: np.ndarray) -> np.ndarray:
        if self.control_model is None or self.treatment_model is None:
            raise ValueError("potential-outcome models are missing")
        control = np.maximum(self.control_model.predict(matrix), 1e-6)
        treatment = self.treatment_model.predict(matrix)
        return (control - treatment) / control

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=float)
        direct = self.direct_model.predict(matrix) if self.direct_model else np.zeros(len(matrix))
        if self.formulation == "C0_direct_paired":
            return direct
        if self.formulation == "C1_t_learner":
            return self._potential_effect(matrix)
        if self.formulation == "C2_s_learner":
            if self.s_model is None:
                raise ValueError("S-learner model is missing")
            control = self.s_model.predict(np.column_stack([matrix, np.zeros(len(matrix))]))
            treatment = self.s_model.predict(np.column_stack([matrix, np.ones(len(matrix))]))
            return (control - treatment) / np.maximum(control, 1e-6)
        potential = self._potential_effect(matrix)
        if self.formulation == "C3_x_learner":
            return 0.5 * (direct + potential)
        residual = self.residual_model.predict(matrix) if self.residual_model else 0.0
        return potential + residual

    def importance(self) -> dict[str, float]:
        return self.direct_model.importance() if self.direct_model else {}

    def to_dict(self) -> dict[str, Any]:
        def dump(model):
            return model.to_dict() if model else None

        return {
            "formulation": self.formulation,
            "kind": self.kind,
            "feature_names": list(self.feature_names),
            "direct_model": dump(self.direct_model),
            "control_model": dump(self.control_model),
            "treatment_model": dump(self.treatment_model),
            "s_model": dump(self.s_model),
            "residual_model": dump(self.residual_model),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CausalEffectLearner":
        def load(model):
            return RegressionModel.from_dict(model) if model else None

        return cls(
            str(value["formulation"]), str(value["kind"]), tuple(value["feature_names"]),
            load(value.get("direct_model")), load(value.get("control_model")),
            load(value.get("treatment_model")), load(value.get("s_model")),
            load(value.get("residual_model")),
        )

