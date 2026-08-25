from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from concordia.errors import ValidationError


class LinUCBPreferenceLearner:
    """Small contextual-bandit baseline for learning route acceptance utility."""

    def __init__(self, dimension: int, alpha: float = 1.0, ridge: float = 1.0) -> None:
        if dimension < 1 or alpha < 0 or ridge <= 0:
            raise ValidationError("invalid LinUCB dimensions or regularization")
        self.dimension = dimension
        self.alpha = alpha
        self._a = np.eye(dimension, dtype=float) * ridge
        self._b = np.zeros(dimension, dtype=float)

    @staticmethod
    def _array(features: Sequence[float], dimension: int) -> np.ndarray:
        vector = np.asarray(features, dtype=float)
        if vector.shape != (dimension,) or not np.all(np.isfinite(vector)):
            raise ValidationError(f"expected {dimension} finite contextual features")
        return vector

    @property
    def estimate(self) -> np.ndarray:
        return np.linalg.solve(self._a, self._b)

    def score(self, features: Sequence[float]) -> float:
        vector = self._array(features, self.dimension)
        inverse_projection = np.linalg.solve(self._a, vector)
        uncertainty = float(np.sqrt(vector @ inverse_projection))
        return float(self.estimate @ vector + self.alpha * uncertainty)

    def choose(self, candidates: Mapping[str, Sequence[float]]) -> str:
        if not candidates:
            raise ValidationError("LinUCB requires at least one candidate")
        scores = {key: self.score(features) for key, features in candidates.items()}
        return max(sorted(scores), key=scores.__getitem__)

    def update(self, features: Sequence[float], reward: float) -> None:
        vector = self._array(features, self.dimension)
        if not np.isfinite(reward):
            raise ValidationError("bandit reward must be finite")
        self._a += np.outer(vector, vector)
        self._b += float(reward) * vector
