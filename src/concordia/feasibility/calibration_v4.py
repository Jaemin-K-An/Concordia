from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -35.0, 35.0)))


def calibration_error(labels: Sequence[int], probabilities: Sequence[float]) -> dict:
    y = np.asarray(labels, dtype=int)
    p = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    curve = []
    for lower, upper in zip(bins[:-1], bins[1:]):
        members = (p >= lower) & (p <= upper if upper == 1.0 else p < upper)
        if members.any():
            weight = float(members.mean())
            predicted = float(p[members].mean())
            observed = float(y[members].mean())
            ece += weight * abs(predicted - observed)
            curve.append(
                {
                    "lower": float(lower),
                    "upper": float(upper),
                    "count": int(members.sum()),
                    "mean_probability": predicted,
                    "observed_rate": observed,
                }
            )
    high = (p >= 0.75) & (p < 0.85)
    return {
        "ece": float(ece),
        "brier_score": float(np.mean((p - y) ** 2)),
        "high_probability_around_0_8_count": int(high.sum()),
        "high_probability_around_0_8_observed_rate": float(y[high].mean()) if high.any() else None,
        "curve": curve,
    }


@dataclass
class ProbabilityCalibrator:
    method: str
    parameters: dict[str, Any] | None = None

    def fit(self, probabilities: Sequence[float], labels: Sequence[int]) -> "ProbabilityCalibrator":
        p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6)
        y = np.asarray(labels, dtype=float)
        if len(p) != len(y) or len(np.unique(y)) < 2:
            raise ValueError("calibration requires aligned probabilities with two outcome classes")
        if self.method == "raw":
            self.parameters = {}
        elif self.method == "platt":
            x = np.log(p / (1.0 - p))
            coefficient = 1.0
            intercept = math.log((y.mean() + 1e-4) / (1.0 - y.mean() + 1e-4))
            for _ in range(1600):
                prediction = _sigmoid(coefficient * x + intercept)
                error = prediction - y
                coefficient = max(0.0, coefficient - 0.03 * float(np.mean(error * x)))
                intercept -= 0.03 * float(error.mean())
            self.parameters = {"coefficient": coefficient, "intercept": intercept}
        elif self.method == "beta":
            matrix = np.column_stack([np.log(p), -np.log(1.0 - p)])
            weights = np.ones(2, dtype=float)
            intercept = 0.0
            for _ in range(2000):
                prediction = _sigmoid(matrix @ weights + intercept)
                error = prediction - y
                weights -= 0.02 * (matrix.T @ error / len(matrix) + 1e-3 * weights)
                weights = np.maximum(weights, 0.0)
                intercept -= 0.02 * float(error.mean())
            self.parameters = {"weights": weights.tolist(), "intercept": intercept}
        elif self.method == "isotonic":
            order = np.argsort(p)
            sorted_p = p[order]
            blocks = [
                {"weight": 1, "sum": float(y[index]), "left": float(value), "right": float(value)}
                for index, value in zip(order, sorted_p)
            ]
            index = 0
            while index < len(blocks) - 1:
                left_mean = blocks[index]["sum"] / blocks[index]["weight"]
                right_mean = blocks[index + 1]["sum"] / blocks[index + 1]["weight"]
                if left_mean <= right_mean + 1e-15:
                    index += 1
                    continue
                blocks[index] = {
                    "weight": blocks[index]["weight"] + blocks[index + 1]["weight"],
                    "sum": blocks[index]["sum"] + blocks[index + 1]["sum"],
                    "left": blocks[index]["left"],
                    "right": blocks[index + 1]["right"],
                }
                blocks.pop(index + 1)
                index = max(0, index - 1)
            self.parameters = {
                "upper_bounds": [block["right"] for block in blocks],
                "values": [block["sum"] / block["weight"] for block in blocks],
            }
        else:
            raise ValueError(f"unknown calibration method: {self.method}")
        return self

    def predict(self, probabilities: Sequence[float]) -> np.ndarray:
        if self.parameters is None:
            raise ValueError("calibrator is not fitted")
        p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6)
        if self.method == "raw":
            return p
        if self.method == "platt":
            logits = np.log(p / (1.0 - p))
            return _sigmoid(
                self.parameters["coefficient"] * logits + self.parameters["intercept"]
            )
        if self.method == "beta":
            matrix = np.column_stack([np.log(p), -np.log(1.0 - p)])
            return _sigmoid(
                matrix @ np.asarray(self.parameters["weights"]) + self.parameters["intercept"]
            )
        bounds = np.asarray(self.parameters["upper_bounds"])
        values = np.asarray(self.parameters["values"])
        indices = np.searchsorted(bounds, p, side="left")
        return values[np.clip(indices, 0, len(values) - 1)]

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method, "parameters": self.parameters}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProbabilityCalibrator":
        return cls(str(value["method"]), dict(value["parameters"]))
