from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional

from concordia.behavior.types import AcceptanceOutcome, RecommendationDecision, RouteOffer
from concordia.errors import ValidationError


@dataclass(frozen=True)
class AcceptanceCoefficients:
    intercept: float = 0.5
    preference_slack: float = -8.0
    utility_gain: float = 3.0
    eta_gain_minutes: float = 0.08
    reliability_gain_minutes2: float = 0.04
    network_benefit: float = 0.0
    source: str = "synthetic_assumption"

    def __post_init__(self) -> None:
        if not self.source:
            raise ValidationError("acceptance coefficients require calibration provenance")
        values = (
            self.intercept,
            self.preference_slack,
            self.utility_gain,
            self.eta_gain_minutes,
            self.reliability_gain_minutes2,
            self.network_benefit,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValidationError("acceptance coefficients must be finite")


class AcceptanceModel:
    """Synthetic or calibrated logistic route-offer acceptance model."""

    def __init__(self, coefficients: Optional[AcceptanceCoefficients] = None) -> None:
        self.coefficients = coefficients or AcceptanceCoefficients()

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0:
            return 1.0 / (1.0 + math.exp(-value))
        exponential = math.exp(value)
        return exponential / (1.0 + exponential)

    def probability(
        self,
        preference_slack: float,
        utility_gain: float,
        eta_gain_minutes: float,
        reliability_gain_minutes2: float,
        network_benefit: float,
    ) -> float:
        if preference_slack < 0:
            raise ValidationError("acceptance input Preference Slack cannot be negative")
        coefficients = self.coefficients
        score = (
            coefficients.intercept
            + coefficients.preference_slack * preference_slack
            + coefficients.utility_gain * utility_gain
            + coefficients.eta_gain_minutes * eta_gain_minutes
            + coefficients.reliability_gain_minutes2 * reliability_gain_minutes2
            + coefficients.network_benefit * network_benefit
        )
        return self._sigmoid(score)

    def decide(
        self,
        offer: RouteOffer,
        rng: random.Random,
        decided_at_seconds: float,
        eligible: bool = True,
    ) -> RecommendationDecision:
        sample = rng.random()
        if not eligible:
            outcome = AcceptanceOutcome.INELIGIBLE
            reason = "user_not_recommendation_eligible"
        elif sample <= offer.predicted_acceptance_probability:
            outcome = AcceptanceOutcome.ACCEPTED
            reason = "sample_within_truthful_acceptance_probability"
        else:
            outcome = AcceptanceOutcome.REJECTED
            reason = "user_rejected_offer"
        return RecommendationDecision(
            offer=offer,
            outcome=outcome,
            sampled_probability=sample,
            decided_at_seconds=decided_at_seconds,
            reason=reason,
        )
