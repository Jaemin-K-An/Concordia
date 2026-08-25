from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from concordia.feasibility.models import FeasibilityModel


@dataclass
class MicroscopicSafetyVeto:
    models: tuple[FeasibilityModel, ...]
    probability_threshold: float
    upper_z: float = 1.645

    @classmethod
    def fit(
        cls,
        matrix: np.ndarray,
        unsafe: Sequence[int],
        feature_names: Sequence[str],
        *,
        seed: int,
        ensemble_size: int = 15,
    ) -> "MicroscopicSafetyVeto":
        x = np.asarray(matrix, dtype=float)
        y = np.asarray(unsafe, dtype=int)
        rng = np.random.default_rng(seed)
        models = []
        for index in range(ensemble_size):
            sample = rng.integers(0, len(x), len(x))
            if len(np.unique(y[sample])) < 2:
                sample = np.arange(len(x))
            model = FeasibilityModel(
                f"micro_safety_{index}",
                "logistic",
                tuple(feature_names),
                regularization=0.10,
                iterations=1800,
            ).fit(x[sample], y[sample])
            models.append(model)
        return cls(tuple(models), 0.10)

    def predict(self, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.vstack([model.predict_proba(matrix) for model in self.models])
        mean = values.mean(axis=0)
        upper = np.clip(mean + self.upper_z * values.std(axis=0), 0.0, 1.0)
        return mean, upper

    def is_safe(self, matrix: np.ndarray) -> np.ndarray:
        _mean, upper = self.predict(matrix)
        return upper < self.probability_threshold

    def to_dict(self) -> dict[str, Any]:
        return {
            "models": [model.to_dict() for model in self.models],
            "probability_threshold": self.probability_threshold,
            "upper_z": self.upper_z,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MicroscopicSafetyVeto":
        return cls(
            tuple(FeasibilityModel.from_dict(model) for model in value["models"]),
            float(value["probability_threshold"]),
            float(value.get("upper_z", 1.645)),
        )
