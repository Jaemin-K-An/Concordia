from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from concordia.uplift_v7.learners import RegressionModel, build_regression_model


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -35.0, 35.0)))


@dataclass
class SafetyClassifier:
    name: str
    kind: str
    feature_names: tuple[str, ...]
    seed: int = 0
    positive_weight: float = 1.0
    parameters: dict[str, Any] | None = None

    def fit(self, matrix: np.ndarray, labels: Sequence[int]) -> "SafetyClassifier":
        x = np.asarray(matrix, dtype=float)
        y = np.asarray(labels, dtype=float)
        if x.ndim != 2 or len(x) != len(y) or len(np.unique(y)) < 2:
            raise ValueError("safety classification requires aligned two-class data")
        if self.kind == "logistic":
            mean = x.mean(axis=0)
            scale = x.std(axis=0)
            scale[scale < 1e-9] = 1.0
            z = np.clip((x - mean) / scale, -12.0, 12.0)
            weights = np.zeros(z.shape[1], dtype=float)
            intercept = float(np.log((y.mean() + 1e-4) / (1.0 - y.mean() + 1e-4)))
            sample_weight = np.where(y > 0.5, self.positive_weight, 1.0)
            step = 0.08 / max(float(np.linalg.norm(z, ord=2) ** 2 / len(z)), 1.0)
            for _ in range(3000):
                prediction = _sigmoid(z @ weights + intercept)
                error = (prediction - y) * sample_weight
                weights -= step * (z.T @ error / len(z) + 0.002 * weights)
                intercept -= step * float(error.mean())
            self.parameters = {
                "mean": mean.tolist(),
                "scale": scale.tolist(),
                "weights": weights.tolist(),
                "intercept": intercept,
            }
            return self
        rng = np.random.default_rng(self.seed)
        indices = np.arange(len(y))
        if self.positive_weight > 1.0:
            positive = indices[y > 0.5]
            extra = int(round((self.positive_weight - 1.0) * len(positive)))
            if extra:
                indices = np.concatenate([indices, rng.choice(positive, extra, replace=True)])
                rng.shuffle(indices)
        model = build_regression_model(self.kind, self.feature_names, self.seed, self.name)
        model.fit(x[indices], y[indices])
        self.parameters = {"regression_model": model.to_dict()}
        return self

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        if self.parameters is None:
            raise ValueError("safety classifier is not fitted")
        x = np.asarray(matrix, dtype=float)
        if self.kind == "logistic":
            z = (x - np.asarray(self.parameters["mean"])) / np.asarray(self.parameters["scale"])
            return _sigmoid(
                np.clip(z, -12.0, 12.0) @ np.asarray(self.parameters["weights"])
                + float(self.parameters["intercept"])
            )
        model = RegressionModel.from_dict(self.parameters["regression_model"])
        return np.clip(model.predict(x), 1e-6, 1.0 - 1e-6)

    def importance(self) -> dict[str, float]:
        if not self.parameters:
            return {name: 0.0 for name in self.feature_names}
        if self.kind == "logistic":
            return {
                name: abs(float(value))
                for name, value in zip(self.feature_names, self.parameters["weights"])
            }
        return RegressionModel.from_dict(self.parameters["regression_model"]).importance()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "feature_names": list(self.feature_names),
            "seed": self.seed,
            "positive_weight": self.positive_weight,
            "parameters": self.parameters,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SafetyClassifier":
        return cls(
            str(value["name"]),
            str(value["kind"]),
            tuple(value["feature_names"]),
            int(value.get("seed", 0)),
            float(value.get("positive_weight", 1.0)),
            dict(value["parameters"]),
        )

