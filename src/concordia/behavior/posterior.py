from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from concordia.errors import ValidationError
from concordia.models import FEATURE_NAMES, PreferenceVector


@dataclass(frozen=True)
class PopulationPrior:
    mean: Tuple[float, ...]
    covariance: Tuple[Tuple[float, ...], ...]
    source: str

    def __post_init__(self) -> None:
        dimension = len(FEATURE_NAMES)
        if len(self.mean) != dimension or len(self.covariance) != dimension or not self.source:
            raise ValidationError("population prior dimension/provenance is invalid")
        matrix = np.asarray(self.covariance, dtype=float)
        if matrix.shape != (dimension, dimension) or np.any(np.linalg.eigvalsh(matrix) <= 0):
            raise ValidationError("population prior covariance must be positive definite")
        if any(value < 0 for value in self.mean) or not math.isclose(sum(self.mean), 1.0):
            raise ValidationError("population prior mean must be a simplex vector")

    @classmethod
    def synthetic_default(cls) -> "PopulationPrior":
        mean = (0.36, 0.16, 0.10, 0.18, 0.10, 0.10)
        covariance = tuple(
            tuple(0.02 if row == column else 0.0 for column in range(len(mean)))
            for row in range(len(mean))
        )
        return cls(mean, covariance, "synthetic_assumption_not_user_calibration")


class UserPreferencePosterior:
    """Online Laplace-style posterior for pairwise route-choice feedback."""

    def __init__(self, prior: PopulationPrior, forgetting_factor: float = 1.0) -> None:
        if not 0 < forgetting_factor <= 1:
            raise ValidationError("posterior forgetting factor must be in (0, 1]")
        self.prior = prior
        self.mean = np.asarray(prior.mean, dtype=float)
        self.covariance = np.asarray(prior.covariance, dtype=float)
        self.forgetting_factor = forgetting_factor
        self.observations = 0

    @staticmethod
    def _feature_difference(first: Sequence[float], second: Sequence[float]) -> np.ndarray:
        difference = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
        if difference.shape != (len(FEATURE_NAMES),) or not np.all(np.isfinite(difference)):
            raise ValidationError("pairwise features must match the preference dimension")
        return difference

    def update_pairwise(
        self,
        chosen_features: Sequence[float],
        rejected_features: Sequence[float],
        learning_rate: float = 1.0,
    ) -> None:
        if learning_rate <= 0:
            raise ValidationError("posterior learning rate must be positive")
        difference = self._feature_difference(chosen_features, rejected_features)
        score = float(np.clip(self.mean @ difference, -30.0, 30.0))
        probability = 1.0 / (1.0 + math.exp(-score))
        variance_projection = float(difference @ self.covariance @ difference)
        gain = self.covariance @ difference / max(1.0 + variance_projection, 1e-12)
        self.mean += learning_rate * gain * (1.0 - probability)
        self.mean = np.clip(self.mean, 1e-9, None)
        self.mean /= self.mean.sum()
        curvature = probability * (1.0 - probability)
        self.covariance = (
            self.covariance - curvature * np.outer(gain, difference @ self.covariance)
        ) / self.forgetting_factor
        self.covariance = (self.covariance + self.covariance.T) / 2
        self.covariance += np.eye(len(self.mean)) * 1e-9
        self.observations += 1

    def apply_drift(self, multipliers: Sequence[float]) -> None:
        values = np.asarray(multipliers, dtype=float)
        if values.shape != self.mean.shape or np.any(values <= 0):
            raise ValidationError("preference drift multipliers must be positive and dimension-matched")
        self.mean *= values
        self.mean /= self.mean.sum()
        self.covariance /= self.forgetting_factor

    def preference_vector(self) -> PreferenceVector:
        return PreferenceVector(**dict(zip(FEATURE_NAMES, self.mean.tolist())))


class DuelingPreferenceLearner:
    def __init__(self, posterior: UserPreferencePosterior) -> None:
        self.posterior = posterior

    def choose_pair(self, candidates: Sequence[Sequence[float]]) -> Tuple[int, int]:
        if len(candidates) < 2:
            raise ValidationError("dueling learner requires at least two candidates")
        best = None
        for first in range(len(candidates)):
            for second in range(first + 1, len(candidates)):
                difference = UserPreferencePosterior._feature_difference(
                    candidates[first], candidates[second]
                )
                uncertainty = float(difference @ self.posterior.covariance @ difference)
                key = (uncertainty, -first, -second)
                if best is None or key > best[0]:
                    best = (key, first, second)
        return best[1], best[2]

    def update(
        self,
        first_features: Sequence[float],
        second_features: Sequence[float],
        chose_first: bool,
    ) -> None:
        chosen, rejected = (
            (first_features, second_features) if chose_first else (second_features, first_features)
        )
        self.posterior.update_pairwise(chosen, rejected)

