from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from concordia.errors import ValidationError


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -35.0, 35.0)))


@dataclass
class FeasibilityModel:
    name: str
    kind: str
    feature_names: tuple[str, ...]
    feature_indices: tuple[int, ...] = ()
    interaction_pairs: tuple[tuple[int, int], ...] = ()
    regularization: float = 0.01
    iterations: int = 1800
    learning_rate: float = 0.05
    seed: int = 0
    parameters: dict[str, Any] | None = None

    def clone(self, seed: int | None = None) -> "FeasibilityModel":
        cloned = copy.deepcopy(self)
        cloned.parameters = None
        if seed is not None:
            cloned.seed = int(seed)
        return cloned

    def _transform(self, matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=float)
        indices = self.feature_indices or tuple(range(matrix.shape[1]))
        columns = [matrix[:, index] for index in indices]
        columns.extend(matrix[:, left] * matrix[:, right] for left, right in self.interaction_pairs)
        return np.column_stack(columns)

    def fit(self, matrix: np.ndarray, labels: np.ndarray) -> "FeasibilityModel":
        matrix = np.asarray(matrix, dtype=float)
        labels = np.asarray(labels, dtype=float)
        if matrix.ndim != 2 or len(matrix) != len(labels) or len(set(labels.tolist())) < 2:
            raise ValidationError("feasibility model requires a two-class matrix")
        transformed = self._transform(matrix)
        if self.kind == "logistic":
            mean = transformed.mean(axis=0)
            scale = transformed.std(axis=0)
            scale[scale < 1e-10] = 1.0
            values = (transformed - mean) / scale
            weights = np.zeros(values.shape[1], dtype=float)
            bias = math.log((labels.mean() + 1e-4) / (1.0 - labels.mean() + 1e-4))
            for _ in range(self.iterations):
                probabilities = _sigmoid(values @ weights + bias)
                error = probabilities - labels
                weights -= self.learning_rate * (
                    values.T @ error / len(values) + self.regularization * weights
                )
                bias -= self.learning_rate * float(error.mean())
            self.parameters = {
                "mean": mean.tolist(),
                "scale": scale.tolist(),
                "weights": weights.tolist(),
                "bias": float(bias),
            }
        elif self.kind == "forest":
            self.parameters = {"stumps": self._fit_forest(transformed, labels)}
        elif self.kind == "boosting":
            self.parameters = self._fit_boosting(transformed, labels)
        else:
            raise ValidationError(f"unknown feasibility model kind: {self.kind}")
        return self

    def _fit_forest(self, matrix: np.ndarray, labels: np.ndarray) -> list[dict[str, float]]:
        rng = np.random.default_rng(self.seed)
        stumps = []
        for _ in range(120):
            sample = rng.integers(0, len(matrix), len(matrix))
            feature = int(rng.integers(0, matrix.shape[1]))
            values = matrix[sample, feature]
            threshold = float(np.quantile(values, rng.choice([0.25, 0.4, 0.5, 0.6, 0.75])))
            left = labels[sample][values <= threshold]
            right = labels[sample][values > threshold]
            stumps.append(
                {
                    "feature": feature,
                    "threshold": threshold,
                    "left": float((left.sum() + 1.0) / (len(left) + 2.0)),
                    "right": float((right.sum() + 1.0) / (len(right) + 2.0)),
                }
            )
        return stumps

    def _fit_boosting(self, matrix: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
        base = float(math.log((labels.mean() + 1e-4) / (1.0 - labels.mean() + 1e-4)))
        score = np.full(len(labels), base, dtype=float)
        stumps = []
        for _ in range(100):
            residual = labels - _sigmoid(score)
            best = None
            for feature in range(matrix.shape[1]):
                for quantile in (0.2, 0.35, 0.5, 0.65, 0.8):
                    threshold = float(np.quantile(matrix[:, feature], quantile))
                    left_mask = matrix[:, feature] <= threshold
                    if not left_mask.any() or left_mask.all():
                        continue
                    left = float(residual[left_mask].mean())
                    right = float(residual[~left_mask].mean())
                    prediction = np.where(left_mask, left, right)
                    loss = float(np.mean((residual - prediction) ** 2))
                    if best is None or loss < best[0]:
                        best = (loss, feature, threshold, left, right, prediction)
            if best is None:
                break
            _, feature, threshold, left, right, prediction = best
            score += self.learning_rate * prediction
            stumps.append(
                {
                    "feature": int(feature),
                    "threshold": threshold,
                    "left": left,
                    "right": right,
                }
            )
        return {"base": base, "learning_rate": self.learning_rate, "stumps": stumps}

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        if self.parameters is None:
            raise ValidationError("feasibility model has not been fitted")
        transformed = self._transform(np.asarray(matrix, dtype=float))
        if self.kind == "logistic":
            mean = np.asarray(self.parameters["mean"])
            scale = np.asarray(self.parameters["scale"])
            weights = np.asarray(self.parameters["weights"])
            return _sigmoid(((transformed - mean) / scale) @ weights + self.parameters["bias"])
        if self.kind == "forest":
            predictions = []
            for stump in self.parameters["stumps"]:
                predictions.append(
                    np.where(
                        transformed[:, int(stump["feature"])] <= stump["threshold"],
                        stump["left"],
                        stump["right"],
                    )
                )
            return np.mean(predictions, axis=0)
        score = np.full(len(transformed), float(self.parameters["base"]))
        for stump in self.parameters["stumps"]:
            score += float(self.parameters["learning_rate"]) * np.where(
                transformed[:, int(stump["feature"])] <= stump["threshold"],
                stump["left"],
                stump["right"],
            )
        return _sigmoid(score)

    def importance(self) -> dict[str, float]:
        if self.parameters is None:
            return {}
        indices = self.feature_indices or tuple(range(len(self.feature_names)))
        names = [self.feature_names[index] for index in indices]
        names.extend(
            f"{self.feature_names[left]}×{self.feature_names[right]}"
            for left, right in self.interaction_pairs
        )
        if self.kind == "logistic":
            return {
                name: abs(float(weight))
                for name, weight in zip(names, self.parameters["weights"])
            }
        totals = {name: 0.0 for name in names}
        for stump in self.parameters["stumps"]:
            totals[names[int(stump["feature"])]] += abs(
                float(stump["left"]) - float(stump["right"])
            )
        return totals

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "feature_names": list(self.feature_names),
            "feature_indices": list(self.feature_indices),
            "interaction_pairs": [list(pair) for pair in self.interaction_pairs],
            "regularization": self.regularization,
            "iterations": self.iterations,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "parameters": self.parameters,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FeasibilityModel":
        return cls(
            name=str(value["name"]),
            kind=str(value["kind"]),
            feature_names=tuple(value["feature_names"]),
            feature_indices=tuple(value.get("feature_indices", ())),
            interaction_pairs=tuple(tuple(pair) for pair in value.get("interaction_pairs", ())),
            regularization=float(value.get("regularization", 0.01)),
            iterations=int(value.get("iterations", 1800)),
            learning_rate=float(value.get("learning_rate", 0.05)),
            seed=int(value.get("seed", 0)),
            parameters=dict(value["parameters"]) if value.get("parameters") else None,
        )


def build_candidate_models(feature_names: Sequence[str], seed: int) -> list[FeasibilityModel]:
    names = tuple(feature_names)
    aps = names.index("alignment_potential_score")
    heterogeneity = names.index("preference_variance")
    rad = names.index("route_attribute_diversity")
    overlap = names.index("route_overlap")
    demand = names.index("demand")
    vc = names.index("volume_capacity_ratio")
    return [
        FeasibilityModel("M0_APS", "logistic", names, (aps,), regularization=0.0, seed=seed),
        FeasibilityModel("M1_logistic", "logistic", names, seed=seed + 1),
        FeasibilityModel(
            "M2_regularized_interactions",
            "logistic",
            names,
            interaction_pairs=((heterogeneity, rad), (aps, overlap), (demand, vc)),
            regularization=0.03,
            seed=seed + 2,
        ),
        FeasibilityModel("M3_random_stump_forest", "forest", names, seed=seed + 3),
        FeasibilityModel(
            "M4_gradient_boosted_stumps",
            "boosting",
            names,
            learning_rate=0.08,
            seed=seed + 4,
        ),
    ]


def load_model(value: Mapping[str, Any]) -> FeasibilityModel:
    return FeasibilityModel.from_dict(value)
