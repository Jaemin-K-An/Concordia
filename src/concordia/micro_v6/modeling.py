from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from concordia.feasibility.hierarchical import HierarchicalSuccessModel
from concordia.feasibility.models import FeasibilityModel

from .features import MICRO_V6_FEATURE_SCHEMA


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -35.0, 35.0)))


def _tree_predict(tree: Mapping[str, Any], matrix: np.ndarray) -> np.ndarray:
    output = np.empty(len(matrix), dtype=float)
    stack = [(tree, np.arange(len(matrix), dtype=int))]
    while stack:
        node, indices = stack.pop()
        if not len(indices):
            continue
        if "value" in node:
            output[indices] = float(node["value"])
            continue
        feature = int(node["feature"])
        left_mask = matrix[indices, feature] <= float(node["threshold"])
        stack.append((node["left"], indices[left_mask]))
        stack.append((node["right"], indices[~left_mask]))
    return output


def _fit_regression_tree(
    matrix: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    rng: np.random.Generator,
    *,
    depth: int,
    max_depth: int,
    minimum_leaf: int,
    feature_subsample: int | None,
    newton_leaf: bool,
) -> dict[str, Any]:
    denominator = max(float(weights.sum()), 1e-9)
    if newton_leaf:
        leaf_value = float(np.clip(target.sum() / denominator, -3.0, 3.0))
    else:
        leaf_value = float(np.sum(weights * target) / denominator)
    if depth >= max_depth or len(matrix) < 2 * minimum_leaf:
        return {"value": leaf_value}
    features = np.arange(matrix.shape[1])
    if feature_subsample and feature_subsample < len(features):
        features = np.sort(rng.choice(features, feature_subsample, replace=False))
    parent_mean = float(np.sum(weights * target) / denominator)
    parent_loss = float(np.sum(weights * (target - parent_mean) ** 2))
    best = None
    for feature in features:
        values = matrix[:, feature]
        if float(values.max() - values.min()) < 1e-12:
            continue
        for quantile in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            threshold = float(np.quantile(values, quantile))
            left = values <= threshold
            left_count = int(left.sum())
            if left_count < minimum_leaf or len(values) - left_count < minimum_leaf:
                continue
            left_weight = max(float(weights[left].sum()), 1e-9)
            right_weight = max(float(weights[~left].sum()), 1e-9)
            left_mean = float(np.sum(weights[left] * target[left]) / left_weight)
            right_mean = float(np.sum(weights[~left] * target[~left]) / right_weight)
            loss = float(
                np.sum(weights[left] * (target[left] - left_mean) ** 2)
                + np.sum(weights[~left] * (target[~left] - right_mean) ** 2)
            )
            gain = parent_loss - loss
            if best is None or gain > best[0]:
                best = (gain, int(feature), threshold, left)
    if best is None or best[0] <= 1e-10:
        return {"value": leaf_value}
    gain, feature, threshold, left = best
    return {
        "feature": feature,
        "threshold": threshold,
        "gain": gain,
        "left": _fit_regression_tree(
            matrix[left], target[left], weights[left], rng,
            depth=depth + 1, max_depth=max_depth, minimum_leaf=minimum_leaf,
            feature_subsample=feature_subsample, newton_leaf=newton_leaf,
        ),
        "right": _fit_regression_tree(
            matrix[~left], target[~left], weights[~left], rng,
            depth=depth + 1, max_depth=max_depth, minimum_leaf=minimum_leaf,
            feature_subsample=feature_subsample, newton_leaf=newton_leaf,
        ),
    }


@dataclass
class V6TreeModel:
    name: str
    kind: str
    feature_names: tuple[str, ...]
    seed: int = 0
    tree_count: int = 120
    max_depth: int = 3
    minimum_leaf: int = 8
    learning_rate: float = 0.06
    parameters: dict[str, Any] | None = None

    def fit(self, matrix: np.ndarray, labels: np.ndarray) -> "V6TreeModel":
        matrix = np.asarray(matrix, dtype=float)
        labels = np.asarray(labels, dtype=float)
        if len(np.unique(labels)) < 2:
            raise ValueError("v6 tree model requires two classes")
        rng = np.random.default_rng(self.seed)
        if self.kind == "random_forest":
            trees = []
            feature_subsample = max(2, int(math.sqrt(matrix.shape[1])))
            positive_weight = len(labels) / max(2.0 * labels.sum(), 1.0)
            negative_weight = len(labels) / max(2.0 * (len(labels) - labels.sum()), 1.0)
            for _ in range(self.tree_count):
                indices = rng.integers(0, len(matrix), len(matrix))
                sampled_labels = labels[indices]
                weights = np.where(sampled_labels > 0.5, positive_weight, negative_weight)
                tree = _fit_regression_tree(
                    matrix[indices], sampled_labels, weights, rng,
                    depth=0, max_depth=self.max_depth, minimum_leaf=self.minimum_leaf,
                    feature_subsample=feature_subsample, newton_leaf=False,
                )
                trees.append(tree)
            self.parameters = {"trees": trees}
            return self
        base_rate = float(np.clip(labels.mean(), 1e-5, 1.0 - 1e-5))
        base = float(math.log(base_rate / (1.0 - base_rate)))
        score = np.full(len(labels), base, dtype=float)
        trees = []
        for _ in range(self.tree_count):
            probability = _sigmoid(score)
            residual = labels - probability
            hessian = np.maximum(probability * (1.0 - probability), 1e-4)
            sample_count = max(2 * self.minimum_leaf, int(0.85 * len(matrix)))
            indices = rng.choice(len(matrix), sample_count, replace=False)
            tree = _fit_regression_tree(
                matrix[indices], residual[indices], hessian[indices], rng,
                depth=0, max_depth=self.max_depth, minimum_leaf=self.minimum_leaf,
                feature_subsample=None, newton_leaf=True,
            )
            score += self.learning_rate * _tree_predict(tree, matrix)
            trees.append(tree)
        self.parameters = {"base": base, "trees": trees}
        return self

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        if self.parameters is None:
            raise ValueError("v6 tree model is not fitted")
        matrix = np.asarray(matrix, dtype=float)
        if self.kind == "random_forest":
            predictions = [_tree_predict(tree, matrix) for tree in self.parameters["trees"]]
            return np.clip(np.mean(predictions, axis=0), 1e-6, 1.0 - 1e-6)
        score = np.full(len(matrix), float(self.parameters["base"]), dtype=float)
        for tree in self.parameters["trees"]:
            score += self.learning_rate * _tree_predict(tree, matrix)
        return _sigmoid(score)

    def importance(self) -> dict[str, float]:
        totals = {name: 0.0 for name in self.feature_names}
        if not self.parameters:
            return totals
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
            "implementation": "v6_tree_ensemble",
            "name": self.name,
            "kind": self.kind,
            "feature_names": list(self.feature_names),
            "seed": self.seed,
            "tree_count": self.tree_count,
            "max_depth": self.max_depth,
            "minimum_leaf": self.minimum_leaf,
            "learning_rate": self.learning_rate,
            "parameters": self.parameters,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "V6TreeModel":
        return cls(
            str(value["name"]), str(value["kind"]), tuple(value["feature_names"]),
            int(value.get("seed", 0)), int(value.get("tree_count", 120)),
            int(value.get("max_depth", 3)), int(value.get("minimum_leaf", 8)),
            float(value.get("learning_rate", 0.06)), dict(value["parameters"]),
        )


def load_tabular_model(value: Mapping[str, Any]):
    if value.get("implementation") == "v6_tree_ensemble":
        return V6TreeModel.from_dict(value)
    return FeasibilityModel.from_dict(value)


def feature_matrix(
    rows: Sequence[Mapping[str, object]],
    feature_names: Sequence[str] = MICRO_V6_FEATURE_SCHEMA,
) -> np.ndarray:
    return np.asarray(
        [
            [float(row["features_pre_decision"][name]) for name in feature_names]
            for row in rows
        ],
        dtype=float,
    )


def micro_regime(features: Mapping[str, float]) -> str:
    if float(features["topology_signalized"]) >= 0.5:
        return "signalized"
    if float(features["topology_merge"]) >= 0.5:
        return "merge_dominant"
    if (
        float(features["alternative_capacity_ratio"]) < 0.72
        or float(features["volume_capacity_ratio"]) > 1.18
    ):
        return "constrained"
    if float(features["navigation_penetration"]) < 0.75:
        return "partial_control"
    if (
        float(features["flow_instability"]) > 0.30
        or float(features["queue_growth_rate"]) > 0.20
        or float(features["short_horizon_speed_oscillation"]) > 2.5
    ):
        return "high_control_unstable"
    return "high_control_stable"


def row_regimes(rows: Sequence[Mapping[str, object]]) -> list[str]:
    return [micro_regime(row["features_pre_decision"]) for row in rows]


@dataclass
class MicroSuccessPredictor:
    name: str
    strategy: str
    feature_names: tuple[str, ...]
    global_model: Any
    regime_models: dict[str, Any] | None = None
    hierarchical_model: HierarchicalSuccessModel | None = None
    minimum_regime_size: int = 40

    def predict_proba(
        self, matrix: np.ndarray, regimes: Sequence[str] | None = None
    ) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=float)
        if self.strategy == "global":
            return self.global_model.predict_proba(matrix)
        if regimes is None:
            raise ValueError("regime-aware prediction requires regimes")
        if self.strategy == "hierarchical":
            if self.hierarchical_model is None:
                raise ValueError("hierarchical model is missing")
            return self.hierarchical_model.predict_proba(matrix, regimes)
        probability = self.global_model.predict_proba(matrix)
        for regime, model in (self.regime_models or {}).items():
            mask = np.asarray([value == regime for value in regimes], dtype=bool)
            if mask.any():
                probability[mask] = model.predict_proba(matrix[mask])
        return probability

    def importance(self) -> dict[str, float]:
        return self.global_model.importance()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "strategy": self.strategy,
            "feature_names": list(self.feature_names),
            "global_model": self.global_model.to_dict(),
            "regime_models": {
                regime: model.to_dict()
                for regime, model in (self.regime_models or {}).items()
            },
            "hierarchical_model": (
                self.hierarchical_model.to_dict() if self.hierarchical_model else None
            ),
            "minimum_regime_size": self.minimum_regime_size,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MicroSuccessPredictor":
        hierarchical = value.get("hierarchical_model")
        return cls(
            name=str(value["name"]),
            strategy=str(value["strategy"]),
            feature_names=tuple(value["feature_names"]),
            global_model=load_tabular_model(value["global_model"]),
            regime_models={
                str(regime): load_tabular_model(model)
                for regime, model in value.get("regime_models", {}).items()
            },
            hierarchical_model=(
                HierarchicalSuccessModel.from_dict(hierarchical) if hierarchical else None
            ),
            minimum_regime_size=int(value.get("minimum_regime_size", 40)),
        )


def _base_model(
    name: str,
    kind: str,
    feature_names: tuple[str, ...],
    seed: int,
    interactions: bool = False,
) -> Any:
    pairs: tuple[tuple[int, int], ...] = ()
    if interactions:
        pairs = tuple(
            (feature_names.index(left), feature_names.index(right))
            for left, right in (
                ("navigation_penetration", "density_mean"),
                ("navigation_penetration", "flow_instability"),
                ("route_overlap", "volume_capacity_ratio"),
                ("predicted_acceptance", "preference_variance"),
                ("drac_proxy_p95", "speed_differential"),
                ("analytical_success_probability", "queue_growth_rate"),
            )
            if left in feature_names and right in feature_names
        )
    if kind in {"boosting", "forest"}:
        return V6TreeModel(
            name=name,
            kind="gradient_boosting" if kind == "boosting" else "random_forest",
            feature_names=feature_names,
            seed=seed,
            tree_count=100 if kind == "boosting" else 140,
            max_depth=2 if kind == "boosting" else 4,
            minimum_leaf=8,
            learning_rate=0.06,
        )
    return FeasibilityModel(
        name=name,
        kind=kind,
        feature_names=feature_names,
        interaction_pairs=pairs,
        regularization=0.04 if interactions else 0.02,
        iterations=100 if kind == "boosting" else 1800,
        learning_rate=0.08 if kind == "boosting" else 0.05,
        seed=seed,
    )


def fit_candidate(
    name: str,
    matrix: np.ndarray,
    labels: np.ndarray,
    regimes: Sequence[str],
    feature_names: Sequence[str] = MICRO_V6_FEATURE_SCHEMA,
    *,
    seed: int = 0,
    minimum_regime_size: int = 40,
) -> MicroSuccessPredictor:
    names = tuple(feature_names)
    specifications = {
        "M0_logistic": ("global", "logistic", False),
        "M1_interaction_logistic": ("global", "logistic", True),
        "M2_gradient_boosting": ("global", "boosting", False),
        "M3_random_forest": ("global", "forest", False),
        "M4_calibrated_gradient_boosting": ("global", "boosting", False),
        "MR_regime_boosting": ("regime", "boosting", False),
        "MH_hierarchical_logistic": ("hierarchical", "logistic", False),
    }
    strategy, kind, interactions = specifications[name]
    global_model = _base_model(name, kind, names, seed, interactions).fit(matrix, labels)
    if strategy == "regime":
        values = np.asarray(regimes, dtype=object)
        models = {}
        for regime in sorted(set(regimes)):
            mask = values == regime
            if int(mask.sum()) < minimum_regime_size or len(np.unique(labels[mask])) < 2:
                continue
            models[regime] = _base_model(
                f"{name}:{regime}", kind, names, seed + len(models) + 1
            ).fit(matrix[mask], labels[mask])
        return MicroSuccessPredictor(
            name, strategy, names, global_model, models, None, minimum_regime_size
        )
    if strategy == "hierarchical":
        hierarchical = HierarchicalSuccessModel.fit(
            matrix,
            labels,
            regimes,
            names,
            pooling_strength=float(minimum_regime_size),
            kind="logistic",
            name=name,
        )
        return MicroSuccessPredictor(
            name,
            strategy,
            names,
            hierarchical.global_model,
            None,
            hierarchical,
            minimum_regime_size,
        )
    return MicroSuccessPredictor(name, strategy, names, global_model, None, None)


def binary_auc(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    positive = p[y == 1]
    negative = p[y == 0]
    if not len(positive) or not len(negative):
        return 0.5
    comparisons = positive[:, None] - negative[None, :]
    return float(np.mean((comparisons > 0) + 0.5 * (comparisons == 0)))


def average_precision(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    y = np.asarray(labels, dtype=int)
    if int(y.sum()) == 0:
        return 0.0
    order = np.argsort(-np.asarray(probabilities, dtype=float), kind="stable")
    ranked = y[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(np.sum(precision * ranked) / y.sum())


def calibration_slope_intercept(
    labels: Sequence[int], probabilities: Sequence[float]
) -> tuple[float, float]:
    y = np.asarray(labels, dtype=float)
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6)
    x = np.log(p / (1.0 - p))
    slope = 1.0
    intercept = 0.0
    for _ in range(2000):
        prediction = 1.0 / (1.0 + np.exp(-np.clip(slope * x + intercept, -35.0, 35.0)))
        error = prediction - y
        slope -= 0.02 * float(np.mean(error * x))
        intercept -= 0.02 * float(error.mean())
    return float(slope), float(intercept)
