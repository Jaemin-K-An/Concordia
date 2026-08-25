from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping, Optional

from concordia.errors import ValidationError
from concordia.models import FEATURE_NAMES, FeatureScales, PreferenceVector, Route


class UtilityModel:
    """Normalized linear private utility with truthful, observable route features."""

    def __init__(self, scales: Optional[FeatureScales] = None) -> None:
        self.scales = scales or FeatureScales()

    def utility(self, preferences: PreferenceVector, route: Route) -> float:
        weights = preferences.normalized().as_dict()
        features = route.features.as_dict()
        scales = self.scales.as_dict()
        disutility = sum(
            weights[name] * features[name] / scales[name]
            for name in FEATURE_NAMES
            if name != "familiarity"
        )
        return -disutility + weights["familiarity"] * features["familiarity"] / scales["familiarity"]

    def utilities(
        self, preferences: PreferenceVector, routes: Iterable[Route]
    ) -> Dict[str, float]:
        result = {route.route_id: self.utility(preferences, route) for route in routes}
        if not result:
            raise ValidationError("at least one candidate route is required")
        return result

    def best_route(self, preferences: PreferenceVector, routes: Iterable[Route]) -> str:
        utilities = self.utilities(preferences, routes)
        return max(sorted(utilities), key=utilities.__getitem__)


def preference_slack(utilities: Mapping[str, float], tolerance: float = 1e-12) -> Dict[str, float]:
    if not utilities:
        raise ValidationError("cannot compute slack for an empty route set")
    best = max(utilities.values())
    result = {route_id: max(0.0, best - value) for route_id, value in utilities.items()}
    if min(result.values()) > tolerance:
        raise ValidationError("at least one best route must have zero preference slack")
    return result


def route_choice_probabilities(
    utilities: Mapping[str, float], rationality: float
) -> Dict[str, float]:
    if not utilities:
        raise ValidationError("cannot choose from an empty route set")
    if rationality < 0:
        raise ValidationError("rationality must be non-negative")
    logits = {route_id: rationality * value for route_id, value in utilities.items()}
    offset = max(logits.values())
    exps = {route_id: math.exp(value - offset) for route_id, value in logits.items()}
    denominator = sum(exps.values())
    probabilities = {route_id: value / denominator for route_id, value in exps.items()}
    if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-12):
        raise ValidationError("route probabilities failed to normalize")
    return probabilities
