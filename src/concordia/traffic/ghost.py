from __future__ import annotations

import math
from dataclasses import dataclass

from concordia.errors import ValidationError


@dataclass(frozen=True)
class GhostRiskModel:
    """Interpretable analytical risk proxy; not an observed phantom-jam detector."""

    intercept: float = -4.0
    saturation_weight: float = 4.0
    speed_cv_weight: float = 2.0
    acceleration_variance_weight: float = 0.5

    def probability(
        self, saturation: float, speed_cv: float, acceleration_variance: float
    ) -> float:
        if saturation < 0 or speed_cv < 0 or acceleration_variance < 0:
            raise ValidationError("ghost-risk inputs cannot be negative")
        logit = (
            self.intercept
            + self.saturation_weight * saturation
            + self.speed_cv_weight * speed_cv
            + self.acceleration_variance_weight * acceleration_variance
        )
        if logit >= 0:
            return 1.0 / (1.0 + math.exp(-logit))
        exponential = math.exp(logit)
        return exponential / (1.0 + exponential)
