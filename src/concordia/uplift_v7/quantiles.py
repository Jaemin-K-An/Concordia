from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .learners import RegressionModel, build_regression_model
from .treatment_effect import CausalEffectLearner


@dataclass
class PinballQuantileRegressor:
    quantile: float
    feature_names: tuple[str, ...]
    regularization: float = 0.001
    iterations: int = 1500
    learning_rate: float = 0.02
    parameters: dict[str, Any] | None = None

    def fit(self, matrix: np.ndarray, target: np.ndarray) -> "PinballQuantileRegressor":
        matrix = np.asarray(matrix, dtype=float)
        target = np.asarray(target, dtype=float)
        mean = matrix.mean(axis=0)
        scale = matrix.std(axis=0)
        scale[scale < 1e-10] = 1.0
        values = (matrix - mean) / scale
        weights = np.zeros(values.shape[1], dtype=float)
        intercept = float(np.quantile(target, self.quantile))
        for iteration in range(self.iterations):
            residual = target - (values @ weights + intercept)
            score = np.where(residual >= 0.0, self.quantile, self.quantile - 1.0)
            rate = self.learning_rate / np.sqrt(1.0 + iteration / 250.0)
            weights += rate * (
                values.T @ score / len(values) - self.regularization * weights
            )
            intercept += rate * float(score.mean())
        self.parameters = {
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "weights": weights.tolist(),
            "intercept": intercept,
        }
        return self

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        if self.parameters is None:
            raise ValueError("quantile regressor is not fitted")
        matrix = np.asarray(matrix, dtype=float)
        values = (matrix - np.asarray(self.parameters["mean"])) / np.asarray(
            self.parameters["scale"]
        )
        return values @ np.asarray(self.parameters["weights"]) + float(
            self.parameters["intercept"]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "quantile": self.quantile,
            "feature_names": list(self.feature_names),
            "regularization": self.regularization,
            "iterations": self.iterations,
            "learning_rate": self.learning_rate,
            "parameters": self.parameters,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PinballQuantileRegressor":
        return cls(
            float(value["quantile"]),
            tuple(value["feature_names"]),
            float(value.get("regularization", 0.001)),
            int(value.get("iterations", 1500)),
            float(value.get("learning_rate", 0.02)),
            dict(value["parameters"]),
        )


@dataclass
class BootstrapCausalEnsemble:
    members: list[CausalEffectLearner]

    @classmethod
    def fit(
        cls,
        formulation: str,
        kind: str,
        matrix: np.ndarray,
        tau: np.ndarray,
        ttt_b1: np.ndarray,
        ttt_adaptive: np.ndarray,
        generated: np.ndarray,
        feature_names: Sequence[str],
        *,
        member_count: int,
        seed: int,
    ) -> "BootstrapCausalEnsemble":
        rng = np.random.default_rng(seed)
        members = []
        for index in range(member_count):
            sample = rng.integers(0, len(matrix), len(matrix))
            members.append(
                CausalEffectLearner.fit(
                    formulation, kind, matrix[sample], tau[sample], ttt_b1[sample],
                    ttt_adaptive[sample], generated[sample], feature_names,
                    seed=seed + 100 * (index + 1),
                )
            )
        return cls(members)

    def prediction_matrix(self, matrix: np.ndarray) -> np.ndarray:
        return np.vstack([member.predict(matrix) for member in self.members])

    def interval(
        self, matrix: np.ndarray, lower_quantile: float, upper_quantile: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = self.prediction_matrix(matrix)
        return (
            values.mean(axis=0),
            np.quantile(values, lower_quantile, axis=0),
            np.quantile(values, upper_quantile, axis=0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"members": [member.to_dict() for member in self.members]}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BootstrapCausalEnsemble":
        return cls([CausalEffectLearner.from_dict(item) for item in value["members"]])


@dataclass
class BootstrapRegressionEnsemble:
    members: list[RegressionModel]

    @classmethod
    def fit(
        cls,
        kind: str,
        matrix: np.ndarray,
        target: np.ndarray,
        feature_names: Sequence[str],
        *,
        member_count: int,
        seed: int,
        name: str,
    ) -> "BootstrapRegressionEnsemble":
        rng = np.random.default_rng(seed)
        names = tuple(feature_names)
        members = []
        for index in range(member_count):
            sample = rng.integers(0, len(matrix), len(matrix))
            members.append(
                build_regression_model(kind, names, seed + 100 * (index + 1), name).fit(
                    matrix[sample], target[sample]
                )
            )
        return cls(members)

    def prediction_matrix(self, matrix: np.ndarray) -> np.ndarray:
        return np.vstack([member.predict(matrix) for member in self.members])

    def interval(
        self, matrix: np.ndarray, lower_quantile: float, upper_quantile: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = self.prediction_matrix(matrix)
        return (
            values.mean(axis=0),
            np.quantile(values, lower_quantile, axis=0),
            np.quantile(values, upper_quantile, axis=0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"members": [member.to_dict() for member in self.members]}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BootstrapRegressionEnsemble":
        return cls([RegressionModel.from_dict(item) for item in value["members"]])
