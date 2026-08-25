from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from concordia.feasibility.models import FeasibilityModel


def _logit(value: float) -> float:
    clipped = min(max(value, 1e-5), 1.0 - 1e-5)
    return math.log(clipped / (1.0 - clipped))


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -35.0, 35.0)))


@dataclass
class HierarchicalSuccessModel:
    global_model: FeasibilityModel
    regime_offsets: dict[str, float]
    pooling_strength: float

    @classmethod
    def fit(
        cls,
        matrix: np.ndarray,
        labels: np.ndarray,
        regimes: Sequence[str],
        feature_names: Sequence[str],
        *,
        pooling_strength: float = 30.0,
        kind: str = "logistic",
        name: str = "M4_hierarchical",
    ) -> "HierarchicalSuccessModel":
        model = FeasibilityModel(
            name,
            kind,
            tuple(feature_names),
            regularization=0.03,
            iterations=500 if kind == "boosting" else 1800,
            learning_rate=0.04,
        ).fit(matrix, labels)
        probability = model.predict_proba(matrix)
        offsets = {}
        regime_values = np.asarray(list(regimes), dtype=object)
        for regime in sorted(set(regimes)):
            mask = regime_values == regime
            count = int(mask.sum())
            observed = (float(labels[mask].sum()) + 1.0) / (count + 2.0)
            expected = float(np.mean(probability[mask]))
            weight = count / (count + pooling_strength)
            offsets[regime] = weight * (_logit(observed) - _logit(expected))
        return cls(model, offsets, pooling_strength)

    def predict_proba(self, matrix: np.ndarray, regimes: Sequence[str]) -> np.ndarray:
        base = np.clip(self.global_model.predict_proba(matrix), 1e-8, 1.0 - 1e-8)
        logits = np.log(base / (1.0 - base))
        offsets = np.asarray([self.regime_offsets.get(regime, 0.0) for regime in regimes])
        return _sigmoid(logits + offsets)

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_model": self.global_model.to_dict(),
            "regime_offsets": self.regime_offsets,
            "pooling_strength": self.pooling_strength,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HierarchicalSuccessModel":
        return cls(
            FeasibilityModel.from_dict(value["global_model"]),
            {str(key): float(item) for key, item in value["regime_offsets"].items()},
            float(value["pooling_strength"]),
        )
