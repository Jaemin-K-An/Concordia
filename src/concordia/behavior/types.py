from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Tuple

from concordia.errors import ValidationError


class AcceptanceOutcome(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    INELIGIBLE = "ineligible"


@dataclass(frozen=True)
class RouteOffer:
    """A truthful route offer; creating it never mutates a simulator route."""

    offer_id: str
    user_id: str
    current_route_id: str
    candidate_route_id: str
    executable_edge_ids: Tuple[str, ...]
    expected_eta_minutes: float
    eta_variance_minutes2: float
    monetary_cost: float
    safety_risk: float
    complexity: float
    familiarity: float
    estimated_utility: float
    reference_utility: float
    preference_slack: float
    network_marginal_benefit_vehicle_minutes: float
    predicted_acceptance_probability: float
    timestamp_seconds: float
    model_version: str
    coefficient_source: str

    def __post_init__(self) -> None:
        if not all(
            (
                self.offer_id,
                self.user_id,
                self.current_route_id,
                self.candidate_route_id,
                self.executable_edge_ids,
                self.model_version,
                self.coefficient_source,
            )
        ):
            raise ValidationError("route offers require identifiers, executable edges, and provenance")
        nonnegative = (
            self.expected_eta_minutes,
            self.eta_variance_minutes2,
            self.monetary_cost,
            self.safety_risk,
            self.complexity,
            self.familiarity,
            self.preference_slack,
            self.timestamp_seconds,
        )
        if any(not math.isfinite(value) or value < 0 for value in nonnegative):
            raise ValidationError("route-offer attributes must be finite and non-negative")
        if not 0 <= self.predicted_acceptance_probability <= 1:
            raise ValidationError("predicted acceptance probability must be in [0, 1]")

    @property
    def utility_gain(self) -> float:
        return self.estimated_utility - self.reference_utility


@dataclass(frozen=True)
class RecommendationDecision:
    offer: RouteOffer
    outcome: AcceptanceOutcome
    sampled_probability: float
    decided_at_seconds: float
    reason: str

    def __post_init__(self) -> None:
        if not 0 <= self.sampled_probability <= 1:
            raise ValidationError("decision sample must be in [0, 1]")
        if self.decided_at_seconds < self.offer.timestamp_seconds or not self.reason:
            raise ValidationError("decision time/reason is invalid")

    @property
    def accepted(self) -> bool:
        return self.outcome is AcceptanceOutcome.ACCEPTED

