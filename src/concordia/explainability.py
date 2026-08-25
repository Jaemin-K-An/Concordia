from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

from concordia.models import Route


@dataclass(frozen=True)
class RecommendationExplanation:
    selected_route: str
    reference_route: str
    eta_delta_minutes: float
    variability_delta_minutes: float
    cost_delta: float
    risk_delta: float
    complexity_delta: float
    familiarity_delta: float
    utility_delta: float
    estimated_network_benefit_vehicle_minutes: float

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def explain_recommendation(
    selected: Route,
    reference: Route,
    selected_utility: float,
    reference_utility: float,
    estimated_network_benefit: float,
) -> RecommendationExplanation:
    """Build a numeric explanation exclusively from supplied, computed values."""
    return RecommendationExplanation(
        selected_route=selected.route_id,
        reference_route=reference.route_id,
        eta_delta_minutes=selected.features.time - reference.features.time,
        variability_delta_minutes=selected.features.variability - reference.features.variability,
        cost_delta=selected.features.cost - reference.features.cost,
        risk_delta=selected.features.risk - reference.features.risk,
        complexity_delta=selected.features.complexity - reference.features.complexity,
        familiarity_delta=selected.features.familiarity - reference.features.familiarity,
        utility_delta=selected_utility - reference_utility,
        estimated_network_benefit_vehicle_minutes=estimated_network_benefit,
    )
