from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from concordia.feasibility.models import FeasibilityModel


@dataclass
class BootstrapFeasibilityEnsemble:
    models: list[FeasibilityModel]

    @classmethod
    def fit(
        cls,
        prototype: FeasibilityModel,
        matrix: np.ndarray,
        labels: np.ndarray,
        groups: Sequence[str],
        *,
        ensemble_size: int,
        seed: int,
    ) -> "BootstrapFeasibilityEnsemble":
        rng = np.random.default_rng(seed)
        groups = np.asarray(groups)
        unique = np.unique(groups)
        models = []
        attempts = 0
        while len(models) < ensemble_size and attempts < ensemble_size * 20:
            attempts += 1
            selected_groups = rng.choice(unique, len(unique), replace=True)
            indices = np.concatenate([np.flatnonzero(groups == group) for group in selected_groups])
            if len(np.unique(labels[indices])) < 2:
                continue
            models.append(prototype.clone(seed + attempts).fit(matrix[indices], labels[indices]))
        if not models:
            models.append(prototype.clone(seed).fit(matrix, labels))
        return cls(models)

    def predict(self, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        predictions = np.asarray([model.predict_proba(matrix) for model in self.models])
        mean = predictions.mean(axis=0)
        uncertainty = predictions.std(axis=0)
        lower = np.quantile(predictions, 0.10, axis=0)
        return mean, uncertainty, lower

    def to_dict(self) -> dict[str, Any]:
        return {"models": [model.to_dict() for model in self.models]}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BootstrapFeasibilityEnsemble":
        return cls([FeasibilityModel.from_dict(item) for item in value["models"]])
