from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

from concordia.adaptive.state import NetworkState, NetworkStateEstimator
from concordia.behavior import AcceptanceModel, RecommendationDecision, RouteOffer
from concordia.errors import ValidationError
from concordia.models import EdgeKey, Route, User
from concordia.network import RoadNetwork
from concordia.optimization import RecedingHorizonOptimizer, RecedingHorizonPlan
from concordia.preferences import UtilityModel
from concordia.simulation import SimulationAdapter


@dataclass(frozen=True)
class ClosedLoopStep:
    timestamp_seconds: float
    state: NetworkState
    plan: RecedingHorizonPlan
    decisions: Tuple[RecommendationDecision, ...]
    accepted_count: int
    rejected_count: int


@dataclass(frozen=True)
class ClosedLoopResult:
    steps: Tuple[ClosedLoopStep, ...]
    final_assignments: Mapping[str, str]

    @property
    def acceptance_rate(self) -> float:
        decisions = [decision for step in self.steps for decision in step.decisions]
        return sum(decision.accepted for decision in decisions) / len(decisions) if decisions else 0.0


class ClosedLoopController:
    """Observe → plan → offer → user decision → accepted-only execution → re-observe."""

    def __init__(
        self,
        network: RoadNetwork,
        routes: Mapping[str, Route],
        users: Sequence[User],
        candidates: Mapping[str, Sequence[str]],
        initial_assignments: Mapping[str, str],
        simulator: SimulationAdapter,
        edge_id_map: Mapping[str, EdgeKey],
        optimizer: RecedingHorizonOptimizer,
        acceptance_model: AcceptanceModel,
        seed: int,
        route_edge_ids: Mapping[str, Sequence[str]],
        model_version: str = "concordia-mpc-v1",
    ) -> None:
        if seed < 0 or set(initial_assignments) != {user.user_id for user in users}:
            raise ValidationError("closed-loop initial assignments/seed are invalid")
        self.network = network
        self.routes = dict(routes)
        self.users = tuple(users)
        self.candidates = candidates
        self.assignments: Dict[str, str] = dict(initial_assignments)
        self.simulator = simulator
        self.edge_id_map = edge_id_map
        self.optimizer = optimizer
        self.acceptance_model = acceptance_model
        self.rng = random.Random(seed)
        self.seed = seed
        self.route_edge_ids = {key: tuple(value) for key, value in route_edge_ids.items()}
        self.model_version = model_version
        self.estimator = NetworkStateEstimator(network)
        self.utility_model = UtilityModel()

    def _offer(
        self,
        user: User,
        plan: RecedingHorizonPlan,
        state: NetworkState,
    ) -> RouteOffer:
        selected_id = plan.first_assignments[user.user_id]
        current_id = self.assignments[user.user_id]
        target_flows = plan.expected_flows
        selected_prediction = self.optimizer.predictor.predict(
            self.routes[selected_id], state, target_flows
        )
        current_prediction = self.optimizer.predictor.predict(
            self.routes[current_id], state, target_flows
        )
        selected_route = Route(
            selected_id,
            self.routes[selected_id].nodes,
            selected_prediction.expected_features,
        )
        current_route = Route(
            current_id,
            self.routes[current_id].nodes,
            current_prediction.expected_features,
        )
        selected_utility = self.utility_model.utility(user.preferences, selected_route)
        current_utility = self.utility_model.utility(user.preferences, current_route)
        features = selected_prediction.expected_features
        return RouteOffer(
            offer_id=f"{self.model_version}:{state.timestamp_seconds:.3f}:{user.user_id}",
            user_id=user.user_id,
            current_route_id=current_id,
            candidate_route_id=selected_id,
            executable_edge_ids=tuple(self.route_edge_ids[selected_id]),
            expected_eta_minutes=features.time,
            eta_variance_minutes2=features.variability,
            monetary_cost=features.cost,
            safety_risk=features.risk,
            complexity=features.complexity,
            familiarity=features.familiarity,
            estimated_utility=selected_utility,
            reference_utility=current_utility,
            preference_slack=plan.dynamic_regrets[user.user_id],
            network_marginal_benefit_vehicle_minutes=max(
                0.0, plan.horizon_objectives[0] - min(plan.horizon_objectives)
            ),
            predicted_acceptance_probability=plan.acceptance_probabilities[user.user_id],
            timestamp_seconds=state.timestamp_seconds,
            model_version=self.model_version,
            coefficient_source=self.acceptance_model.coefficients.source,
        )

    def run(self, steps: int) -> ClosedLoopResult:
        if steps < 1:
            raise ValidationError("closed-loop run requires at least one step")
        self.simulator.start(self.seed)
        records = []
        try:
            for _ in range(steps):
                snapshot = self.simulator.step()
                state = self.estimator.from_snapshot(
                    snapshot,
                    self.edge_id_map,
                    source=type(self.simulator).__name__,
                )
                plan = self.optimizer.plan(
                    state,
                    self.users,
                    self.candidates,
                    self.assignments,
                )
                decisions = []
                for user in self.users:
                    if plan.first_assignments[user.user_id] == self.assignments[user.user_id]:
                        continue
                    offer = self._offer(user, plan, state)
                    decision = self.acceptance_model.decide(
                        offer,
                        self.rng,
                        decided_at_seconds=state.timestamp_seconds,
                    )
                    executed = self.simulator.execute_accepted_route(decision)
                    if executed:
                        self.assignments[user.user_id] = offer.candidate_route_id
                    decisions.append(decision)
                records.append(
                    ClosedLoopStep(
                        timestamp_seconds=state.timestamp_seconds,
                        state=state,
                        plan=plan,
                        decisions=tuple(decisions),
                        accepted_count=sum(decision.accepted for decision in decisions),
                        rejected_count=sum(not decision.accepted for decision in decisions),
                    )
                )
        finally:
            self.simulator.close()
        return ClosedLoopResult(tuple(records), dict(self.assignments))
