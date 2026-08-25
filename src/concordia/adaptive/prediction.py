from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from concordia.adaptive.state import NetworkState
from concordia.errors import ValidationError
from concordia.models import EdgeKey, Route, RouteFeatures, User
from concordia.network import RoadNetwork
from concordia.preferences import UtilityModel, preference_slack
from concordia.traffic import GhostRiskModel


@dataclass(frozen=True)
class DynamicRoutePrediction:
    route: Route
    expected_features: RouteFeatures
    horizon_eta_minutes: Tuple[float, ...]
    eta_standard_deviation_minutes: float
    social_ghost_risk: float


class DynamicRoutePredictor:
    """Deterministic BPR rollout for receding-horizon route attributes."""

    def __init__(
        self,
        network: RoadNetwork,
        horizon_steps: int = 3,
        flow_relaxation: float = 0.5,
    ) -> None:
        if horizon_steps < 1 or not 0 < flow_relaxation <= 1:
            raise ValidationError("prediction horizon/relaxation is invalid")
        self.network = network
        self.horizon_steps = horizon_steps
        self.flow_relaxation = flow_relaxation
        self.ghost_model = GhostRiskModel()

    def project_flows(
        self,
        state: NetworkState,
        target_flows: Mapping[EdgeKey, float],
    ) -> Tuple[Dict[EdgeKey, float], ...]:
        current = dict(state.flows)
        trajectory = []
        for _ in range(self.horizon_steps):
            current = {
                edge: max(
                    0.0,
                    current[edge]
                    + self.flow_relaxation * (float(target_flows.get(edge, 0.0)) - current[edge]),
                )
                for edge in self.network.edges
            }
            trajectory.append(dict(current))
        return tuple(trajectory)

    def predict(
        self,
        route: Route,
        state: NetworkState,
        target_flows: Mapping[EdgeKey, float],
    ) -> DynamicRoutePrediction:
        projected = self.project_flows(state, target_flows)
        features = [self.network.path_features(route.nodes, flows) for flows in projected]
        eta = tuple(item.time for item in features)
        mean_features = RouteFeatures(
            time=float(np.mean(eta)),
            variability=float(np.mean([item.variability for item in features])),
            cost=float(np.mean([item.cost for item in features])),
            risk=float(np.mean([item.risk for item in features])),
            complexity=float(np.mean([item.complexity for item in features])),
            familiarity=route.features.familiarity,
        )
        final_flows = projected[-1]
        ghost_risk = sum(
            self.ghost_model.probability(
                final_flows[edge] / self.network.edge_data(edge).capacity,
                0.0,
                0.0,
            )
            for edge in route.edges
        )
        return DynamicRoutePrediction(
            route=route,
            expected_features=mean_features,
            horizon_eta_minutes=eta,
            eta_standard_deviation_minutes=float(np.std(eta)),
            social_ghost_risk=ghost_risk,
        )


def dynamic_preference_slack(
    user: User,
    predictions: Sequence[DynamicRoutePrediction],
    utility_model: UtilityModel,
) -> Dict[str, float]:
    if not predictions:
        raise ValidationError("dynamic slack requires route predictions")
    dynamic_routes = [
        Route(
            route_id=prediction.route.route_id,
            nodes=prediction.route.nodes,
            features=prediction.expected_features,
        )
        for prediction in predictions
    ]
    utilities = utility_model.utilities(user.preferences, dynamic_routes)
    return preference_slack(utilities)
