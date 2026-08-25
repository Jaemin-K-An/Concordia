from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass
class BenefitModel:
    name: str
    kind: str
    quantile: float | None = None
    penalty: float = 1e-3
    iterations: int = 160
    learning_rate: float = 0.08
    parameters: dict[str, Any] | None = None

    def fit(self, matrix: np.ndarray, target: np.ndarray) -> "BenefitModel":
        x = np.asarray(matrix, dtype=float)
        y = np.asarray(target, dtype=float)
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale < 1e-10] = 1.0
        values = (x - mean) / scale
        if self.kind in {"ridge", "elastic"}:
            design = np.column_stack([np.ones(len(values)), values])
            regularizer = np.eye(design.shape[1]) * self.penalty
            regularizer[0, 0] = 0.0
            coefficients = np.linalg.solve(design.T @ design + regularizer, design.T @ y)
            if self.kind == "elastic":
                coefficients[1:] = np.sign(coefficients[1:]) * np.maximum(
                    0.0, np.abs(coefficients[1:]) - self.penalty
                )
            prediction = design @ coefficients
            self.parameters = {
                "mean": mean.tolist(),
                "scale": scale.tolist(),
                "intercept": float(coefficients[0]),
                "coefficients": coefficients[1:].tolist(),
                "residual_standard_deviation": float(np.std(y - prediction, ddof=1)),
            }
        elif self.kind == "boosting":
            prediction = np.full(
                len(y),
                float(np.quantile(y, self.quantile)) if self.quantile is not None else float(y.mean()),
            )
            base = float(prediction[0])
            stumps = []
            for _ in range(self.iterations):
                gradient = (
                    self.quantile - (y < prediction).astype(float)
                    if self.quantile is not None
                    else y - prediction
                )
                best = None
                for feature in range(values.shape[1]):
                    for threshold in np.unique(
                        np.quantile(values[:, feature], [0.2, 0.35, 0.5, 0.65, 0.8])
                    ):
                        left_mask = values[:, feature] <= threshold
                        if not left_mask.any() or left_mask.all():
                            continue
                        left = float(gradient[left_mask].mean())
                        right = float(gradient[~left_mask].mean())
                        estimate = np.where(left_mask, left, right)
                        loss = float(np.mean((gradient - estimate) ** 2))
                        if best is None or loss < best[0]:
                            best = (loss, feature, float(threshold), left, right, estimate)
                if best is None:
                    break
                _, feature, threshold, left, right, estimate = best
                prediction += self.learning_rate * estimate
                stumps.append(
                    {
                        "feature": int(feature),
                        "threshold": threshold,
                        "left": left,
                        "right": right,
                    }
                )
            self.parameters = {
                "mean": mean.tolist(),
                "scale": scale.tolist(),
                "base": base,
                "learning_rate": self.learning_rate,
                "stumps": stumps,
                "residual_standard_deviation": float(np.std(y - prediction, ddof=1)),
            }
        else:
            raise ValueError(f"unknown benefit model kind: {self.kind}")
        return self

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        if self.parameters is None:
            raise ValueError("benefit model is not fitted")
        x = np.asarray(matrix, dtype=float)
        values = (x - np.asarray(self.parameters["mean"])) / np.asarray(self.parameters["scale"])
        if self.kind in {"ridge", "elastic"}:
            return self.parameters["intercept"] + values @ np.asarray(
                self.parameters["coefficients"]
            )
        prediction = np.full(len(values), float(self.parameters["base"]))
        for stump in self.parameters["stumps"]:
            prediction += self.parameters["learning_rate"] * np.where(
                values[:, int(stump["feature"])] <= stump["threshold"],
                stump["left"],
                stump["right"],
            )
        return prediction

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "quantile": self.quantile,
            "penalty": self.penalty,
            "iterations": self.iterations,
            "learning_rate": self.learning_rate,
            "parameters": self.parameters,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BenefitModel":
        return cls(
            name=str(value["name"]),
            kind=str(value["kind"]),
            quantile=value.get("quantile"),
            penalty=float(value.get("penalty", 1e-3)),
            iterations=int(value.get("iterations", 160)),
            learning_rate=float(value.get("learning_rate", 0.08)),
            parameters=dict(value["parameters"]) if value.get("parameters") else None,
        )


def regression_metrics(target: np.ndarray, prediction: np.ndarray, quantile: float = 0.1) -> dict:
    y = np.asarray(target, dtype=float)
    p = np.asarray(prediction, dtype=float)
    error = y - p
    pinball = np.maximum(quantile * error, (quantile - 1.0) * error)
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "pinball_loss": float(np.mean(pinball)),
        "lower_bound_coverage": float(np.mean(y >= p)),
    }
