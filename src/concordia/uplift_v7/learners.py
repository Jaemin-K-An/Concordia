from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from concordia.micro_v6.modeling import _fit_regression_tree, _tree_predict


@dataclass
class RegressionModel:
    name: str
    kind: str
    feature_names: tuple[str, ...]
    seed: int = 0
    regularization: float = 0.05
    tree_count: int = 100
    max_depth: int = 3
    minimum_leaf: int = 10
    learning_rate: float = 0.06
    parameters: dict[str, Any] | None = None

    def fit(self, matrix: np.ndarray, target: np.ndarray) -> "RegressionModel":
        matrix = np.asarray(matrix, dtype=float)
        target = np.asarray(target, dtype=float)
        if matrix.ndim != 2 or len(matrix) != len(target):
            raise ValueError("regression model requires aligned two-dimensional data")
        rng = np.random.default_rng(self.seed)
        if self.kind in {"ridge", "elastic_net"}:
            mean = matrix.mean(axis=0)
            scale = matrix.std(axis=0)
            scale[scale < 1e-10] = 1.0
            values = (matrix - mean) / scale
            target_mean = float(target.mean())
            centered = target - target_mean
            if self.kind == "ridge":
                gram = values.T @ values / len(values)
                weights = np.linalg.solve(
                    gram + self.regularization * np.eye(values.shape[1]),
                    values.T @ centered / len(values),
                )
            else:
                weights = np.zeros(values.shape[1], dtype=float)
                step = 0.08 / max(float(np.linalg.norm(values, ord=2) ** 2 / len(values)), 1.0)
                for _ in range(2500):
                    gradient = values.T @ (values @ weights - centered) / len(values)
                    candidate = weights - step * gradient
                    penalty = step * self.regularization * 0.5
                    weights = np.sign(candidate) * np.maximum(np.abs(candidate) - penalty, 0.0)
            self.parameters = {
                "mean": mean.tolist(),
                "scale": scale.tolist(),
                "weights": weights.tolist(),
                "intercept": target_mean,
            }
            return self
        if self.kind == "random_forest":
            trees = []
            feature_subsample = max(2, int(math.sqrt(matrix.shape[1])))
            for _ in range(self.tree_count):
                indices = rng.integers(0, len(matrix), len(matrix))
                trees.append(
                    _fit_regression_tree(
                        matrix[indices], target[indices], np.ones(len(indices)), rng,
                        depth=0, max_depth=self.max_depth, minimum_leaf=self.minimum_leaf,
                        feature_subsample=feature_subsample, newton_leaf=False,
                    )
                )
            self.parameters = {"trees": trees}
            return self
        base = float(target.mean())
        prediction = np.full(len(target), base, dtype=float)
        trees = []
        for _ in range(self.tree_count):
            residual = target - prediction
            count = max(2 * self.minimum_leaf, int(0.85 * len(matrix)))
            indices = rng.choice(len(matrix), count, replace=False)
            tree = _fit_regression_tree(
                matrix[indices], residual[indices], np.ones(len(indices)), rng,
                depth=0, max_depth=self.max_depth, minimum_leaf=self.minimum_leaf,
                feature_subsample=None, newton_leaf=False,
            )
            prediction += self.learning_rate * _tree_predict(tree, matrix)
            trees.append(tree)
        self.parameters = {"base": base, "trees": trees}
        return self

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        if self.parameters is None:
            raise ValueError("regression model is not fitted")
        matrix = np.asarray(matrix, dtype=float)
        if self.kind in {"ridge", "elastic_net"}:
            values = (matrix - np.asarray(self.parameters["mean"])) / np.asarray(
                self.parameters["scale"]
            )
            return values @ np.asarray(self.parameters["weights"]) + float(
                self.parameters["intercept"]
            )
        if self.kind == "random_forest":
            return np.mean(
                [_tree_predict(tree, matrix) for tree in self.parameters["trees"]], axis=0
            )
        prediction = np.full(len(matrix), float(self.parameters["base"]), dtype=float)
        for tree in self.parameters["trees"]:
            prediction += self.learning_rate * _tree_predict(tree, matrix)
        return prediction

    def importance(self) -> dict[str, float]:
        totals = {name: 0.0 for name in self.feature_names}
        if not self.parameters:
            return totals
        if self.kind in {"ridge", "elastic_net"}:
            return {
                name: abs(float(value))
                for name, value in zip(self.feature_names, self.parameters["weights"])
            }
        stack = list(self.parameters["trees"])
        while stack:
            node = stack.pop()
            if "value" in node:
                continue
            totals[self.feature_names[int(node["feature"])]] += float(node.get("gain", 0.0))
            stack.extend((node["left"], node["right"]))
        return totals

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "feature_names": list(self.feature_names),
            "seed": self.seed,
            "regularization": self.regularization,
            "tree_count": self.tree_count,
            "max_depth": self.max_depth,
            "minimum_leaf": self.minimum_leaf,
            "learning_rate": self.learning_rate,
            "parameters": self.parameters,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RegressionModel":
        return cls(
            str(value["name"]), str(value["kind"]), tuple(value["feature_names"]),
            int(value.get("seed", 0)), float(value.get("regularization", 0.05)),
            int(value.get("tree_count", 100)), int(value.get("max_depth", 3)),
            int(value.get("minimum_leaf", 10)), float(value.get("learning_rate", 0.06)),
            dict(value["parameters"]),
        )


def build_regression_model(
    kind: str, feature_names: tuple[str, ...], seed: int, name: str
) -> RegressionModel:
    return RegressionModel(
        name,
        kind,
        feature_names,
        seed=seed,
        regularization=0.05 if kind == "ridge" else 0.02,
        tree_count=90 if kind == "gradient_boosting" else 110,
        max_depth=2 if kind == "gradient_boosting" else 4,
        minimum_leaf=10,
        learning_rate=0.06,
    )

