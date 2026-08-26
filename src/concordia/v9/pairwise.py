from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .action_features import ACTION_FEATURE_SCHEMA, ACTION_NUMERIC_FEATURES


PAIRWISE_STATE_FEATURES = (
    "density_mean",
    "flow_mean",
    "occupancy_mean",
    "mean_speed",
    "queue_length",
    "demand_vehicles_per_hour",
    "flow_instability",
    "queue_growth_rate",
    "alternative_capacity_ratio",
    "volume_capacity_ratio",
    "route_length_ratio",
    "topology_merge",
    "topology_signalized",
    "topology_two_route",
    "topology_asymmetric",
    "perturbation_strength",
    "predicted_acceptance",
    "acceptance_multiplier",
    "navigation_penetration",
    "drac_proxy_p95",
)
PAIRWISE_ACTION_FEATURES = (
    *ACTION_FEATURE_SCHEMA,
    *(f"square_{name}" for name in ACTION_NUMERIC_FEATURES),
)


def _action_vector(row: Mapping[str, object]) -> np.ndarray:
    features = row["action_features"]
    base = [float(features[name]) for name in ACTION_FEATURE_SCHEMA]
    squared = [float(features[name]) ** 2 for name in ACTION_NUMERIC_FEATURES]
    return np.asarray([*base, *squared], dtype=float)


def interaction_matrix(rows: Sequence[Mapping[str, object]]) -> np.ndarray:
    output = []
    for row in rows:
        state = np.asarray(
            [float(row["state_features"][name]) for name in PAIRWISE_STATE_FEATURES],
            dtype=float,
        )
        action = _action_vector(row)
        output.append(np.concatenate((action, np.outer(state, action).ravel())))
    return np.asarray(output, dtype=float)


def _safe_value(row: Mapping[str, object]) -> float:
    outcome = row["outcomes"]
    safe = (
        float(outcome["tau_s"]) <= 0.25
        and float(outcome["max_regret"]) <= 0.08
        and bool(outcome["legal"])
    )
    return float(outcome["tau_t_relative"]) if safe else -0.25


@dataclass
class PairwiseActionRanker:
    seed: int = 9200
    learning_rate: float = 0.025
    iterations: int = 1600
    regularization: float = 0.002
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None
    weights: np.ndarray | None = None

    def fit(self, rows: Sequence[Mapping[str, object]]) -> "PairwiseActionRanker":
        matrix = interaction_matrix(rows)
        self.mean = matrix.mean(axis=0)
        self.scale = matrix.std(axis=0)
        self.scale[self.scale < 1e-9] = 1.0
        normalized = np.clip((matrix - self.mean) / self.scale, -12.0, 12.0).astype(np.float32)
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            grouped[str(row["state_id"])].append(index)
        differences = []
        labels = []
        values = [_safe_value(row) for row in rows]
        for indices in grouped.values():
            for offset, left in enumerate(indices):
                for right in indices[offset + 1:]:
                    delta = values[left] - values[right]
                    if abs(delta) <= 1e-12:
                        continue
                    differences.append(normalized[left] - normalized[right])
                    labels.append(float(delta > 0.0))
        pair_matrix = np.asarray(differences, dtype=np.float32)
        target = np.asarray(labels, dtype=np.float32)
        if not len(pair_matrix) or len(np.unique(target)) != 2:
            raise ValueError("pairwise ranker requires non-tied two-class action pairs")
        rng = np.random.default_rng(self.seed)
        weights = np.zeros(pair_matrix.shape[1], dtype=np.float64)
        first = np.zeros_like(weights)
        second = np.zeros_like(weights)
        batch_size = min(768, len(pair_matrix))
        for iteration in range(1, self.iterations + 1):
            indices = rng.choice(len(pair_matrix), batch_size, replace=False)
            batch = pair_matrix[indices].astype(np.float64)
            prediction = 1.0 / (1.0 + np.exp(-np.clip(batch @ weights, -30.0, 30.0)))
            gradient = batch.T @ (prediction - target[indices]) / batch_size
            gradient += self.regularization * weights
            first = 0.9 * first + 0.1 * gradient
            second = 0.999 * second + 0.001 * gradient**2
            first_hat = first / (1.0 - 0.9**iteration)
            second_hat = second / (1.0 - 0.999**iteration)
            weights -= self.learning_rate * first_hat / (np.sqrt(second_hat) + 1e-8)
        self.weights = weights
        return self

    def predict(self, rows: Sequence[Mapping[str, object]]) -> np.ndarray:
        if self.mean is None or self.scale is None or self.weights is None:
            raise ValueError("pairwise ranker is not fitted")
        matrix = interaction_matrix(rows)
        normalized = np.clip((matrix - self.mean) / self.scale, -12.0, 12.0)
        return normalized @ self.weights

    def to_dict(self) -> dict[str, Any]:
        if self.mean is None or self.scale is None or self.weights is None:
            raise ValueError("pairwise ranker is not fitted")
        return {
            "implementation": "v9_pairwise_ranknet_linear_interactions",
            "seed": self.seed,
            "learning_rate": self.learning_rate,
            "iterations": self.iterations,
            "regularization": self.regularization,
            "state_features": list(PAIRWISE_STATE_FEATURES),
            "action_features": list(PAIRWISE_ACTION_FEATURES),
            "interaction_feature_count": len(self.weights),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "weights": self.weights.tolist(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PairwiseActionRanker":
        ranker = cls(
            int(value["seed"]), float(value["learning_rate"]),
            int(value["iterations"]), float(value["regularization"]),
        )
        ranker.mean = np.asarray(value["mean"], dtype=float)
        ranker.scale = np.asarray(value["scale"], dtype=float)
        ranker.weights = np.asarray(value["weights"], dtype=float)
        return ranker
