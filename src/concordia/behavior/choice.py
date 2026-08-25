from __future__ import annotations

import random
from typing import Mapping

from concordia.errors import ValidationError
from concordia.preferences import route_choice_probabilities


class BehavioralChoiceModel:
    """Bounded-rational route choice independent from recommendation acceptance."""

    def probabilities(self, utilities: Mapping[str, float], rationality: float) -> dict[str, float]:
        return route_choice_probabilities(utilities, rationality)

    def sample(self, utilities: Mapping[str, float], rationality: float, rng: random.Random) -> str:
        probabilities = self.probabilities(utilities, rationality)
        draw = rng.random()
        cumulative = 0.0
        for route_id in sorted(probabilities):
            cumulative += probabilities[route_id]
            if draw <= cumulative:
                return route_id
        if not probabilities:
            raise ValidationError("choice model requires at least one route")
        return sorted(probabilities)[-1]

