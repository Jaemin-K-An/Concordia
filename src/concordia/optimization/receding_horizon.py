from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

from concordia.adaptive import DynamicRoutePredictor, NetworkState
from concordia.behavior import AcceptanceModel
from concordia.errors import InfeasibleAssignment, ValidationError
from concordia.models import EdgeKey, Route, User
from concordia.network import RoadNetwork
from concordia.optimization.adaptive import ObjectiveWeights
from concordia.optimization.objective import route_concentration, total_travel_time, upper_cvar
from concordia.preferences import UtilityModel, preference_slack
from concordia.traffic import GhostRiskModel


@dataclass(frozen=True)
class RecedingHorizonPlan:
    """A horizon plan whose `first_assignments` are the only executable action."""

    first_assignments: Mapping[str, str]
    acceptance_probabilities: Mapping[str, float]
    dynamic_regrets: Mapping[str, float]
    horizon_objectives: Tuple[float, ...]
    expected_safety_cvar: float
    baseline_safety_cvar: float
    objective: float
    solve_time_seconds: float
    combinations_evaluated: int
    method: str = "receding_horizon_constant_action_rollout"


class RecedingHorizonOptimizer:
    """Small-instance MPC baseline with stochastic acceptance and hard filters.

    A candidate assignment is held as a target over the prediction horizon only for scoring.
    The controller executes its first action, observes again, and replans.
    """

    def __init__(
        self,
        network: RoadNetwork,
        routes: Mapping[str, Route],
        vehicle_flow: float,
        horizon_steps: int = 3,
        flow_relaxation: float = 0.5,
        discount: float = 0.95,
        minimum_acceptance_probability: float = 0.0,
        safety_delta: float = 0.0,
        weights: Optional[ObjectiveWeights] = None,
        acceptance_model: Optional[AcceptanceModel] = None,
        max_combinations: int = 100_000,
    ) -> None:
        if vehicle_flow <= 0 or not 0 < discount <= 1:
            raise ValidationError("MPC vehicle flow/discount is invalid")
        if not 0 <= minimum_acceptance_probability <= 1 or safety_delta < 0:
            raise ValidationError("MPC acceptance/safety bounds are invalid")
        self.network = network
        self.routes = dict(routes)
        self.vehicle_flow = vehicle_flow
        self.predictor = DynamicRoutePredictor(network, horizon_steps, flow_relaxation)
        self.discount = discount
        self.minimum_acceptance_probability = minimum_acceptance_probability
        self.safety_delta = safety_delta
        self.weights = weights or ObjectiveWeights()
        self.acceptance_model = acceptance_model or AcceptanceModel()
        self.utility_model = UtilityModel()
        self.max_combinations = max_combinations
        self.ghost_model = GhostRiskModel()

    def _flows(self, assignments: Mapping[str, str]) -> Dict[EdgeKey, float]:
        flows = {edge: 0.0 for edge in self.network.edges}
        for route_id in assignments.values():
            for edge in self.routes[route_id].edges:
                flows[edge] += self.vehicle_flow
        return flows

    def _expected_flows(
        self,
        current: Mapping[str, str],
        proposed: Mapping[str, str],
        acceptance: Mapping[str, float],
    ) -> Dict[EdgeKey, float]:
        flows = {edge: 0.0 for edge in self.network.edges}
        for user_id, proposed_route in proposed.items():
            current_route = current[user_id]
            probability = acceptance[user_id]
            for edge in self.routes[proposed_route].edges:
                flows[edge] += self.vehicle_flow * probability
            for edge in self.routes[current_route].edges:
                flows[edge] += self.vehicle_flow * (1.0 - probability)
        return flows

    def plan(
        self,
        state: NetworkState,
        users: Sequence[User],
        candidates: Mapping[str, Sequence[str]],
        current_assignments: Mapping[str, str],
    ) -> RecedingHorizonPlan:
        started = time.perf_counter()
        if set(current_assignments) != {user.user_id for user in users}:
            raise ValidationError("current assignments must cover every user exactly once")
        option_lists = []
        combinations = 1
        for user in users:
            options = tuple(candidates.get(user.user_id, ()))
            if not options or any(route_id not in self.routes for route_id in options):
                raise ValidationError(f"invalid MPC candidates for {user.user_id}")
            combinations *= len(options)
            if combinations > self.max_combinations:
                raise ValidationError("MPC enumeration exceeds configured small-instance limit")
            option_lists.append(options)
        baseline_risks = [
            self.routes[current_assignments[user.user_id]].features.risk for user in users
        ]
        baseline_cvar = upper_cvar(baseline_risks)
        current_ttt = total_travel_time(self.network, state.flows)
        best = None
        evaluated = 0
        for selected in itertools.product(*option_lists):
            proposed = {user.user_id: route_id for user, route_id in zip(users, selected)}
            full_target_flows = self._flows(proposed)
            predictions = {
                route_id: self.predictor.predict(route, state, full_target_flows)
                for route_id, route in self.routes.items()
            }
            acceptance = {}
            regrets = {}
            feasible = True
            for user in users:
                predicted_routes = [
                    Route(
                        route_id=route_id,
                        nodes=self.routes[route_id].nodes,
                        features=predictions[route_id].expected_features,
                    )
                    for route_id in candidates[user.user_id]
                ]
                utilities = self.utility_model.utilities(user.preferences, predicted_routes)
                slacks = preference_slack(utilities)
                selected_route = proposed[user.user_id]
                current_route = current_assignments[user.user_id]
                regret = slacks[selected_route]
                regrets[user.user_id] = regret
                if regret > user.epsilon + 1e-10:
                    feasible = False
                    break
                if selected_route == current_route:
                    probability = 1.0
                else:
                    selected_features = predictions[selected_route].expected_features
                    current_features = predictions[current_route].expected_features
                    probability = self.acceptance_model.probability(
                        preference_slack=regret,
                        utility_gain=utilities[selected_route] - utilities[current_route],
                        eta_gain_minutes=current_features.time - selected_features.time,
                        reliability_gain_minutes2=(
                            current_features.variability - selected_features.variability
                        ),
                        network_benefit=max(0.0, current_ttt - total_travel_time(self.network, full_target_flows)),
                    )
                if probability < self.minimum_acceptance_probability:
                    feasible = False
                    break
                acceptance[user.user_id] = probability
            if not feasible:
                continue
            expected_risks = []
            for user in users:
                user_id = user.user_id
                selected_risk = self.routes[proposed[user_id]].features.risk
                current_risk = self.routes[current_assignments[user_id]].features.risk
                probability = acceptance[user_id]
                expected_risks.append(probability * selected_risk + (1 - probability) * current_risk)
            safety_cvar = upper_cvar(expected_risks)
            if safety_cvar > baseline_cvar + self.safety_delta + 1e-10:
                continue
            expected_target = self._expected_flows(current_assignments, proposed, acceptance)
            trajectory = self.predictor.project_flows(state, expected_target)
            objectives = []
            hhi, _ = route_concentration(proposed)
            for flows in trajectory:
                ttt = total_travel_time(self.network, flows)
                ghost = sum(
                    self.ghost_model.probability(
                        flows[edge] / self.network.edge_data(edge).capacity,
                        0.0,
                        0.0,
                    )
                    * flows[edge]
                    for edge in self.network.edges
                )
                objectives.append(
                    ttt
                    + self.weights.ghost_risk * ghost
                    + self.weights.safety_risk * sum(expected_risks)
                    + self.weights.concentration * hhi
                )
            objective = sum(
                (self.discount**index) * value for index, value in enumerate(objectives)
            )
            evaluated += 1
            key = (objective, tuple(sorted(proposed.items())))
            if best is None or key < best[0]:
                best = (key, proposed, acceptance, regrets, objectives, safety_cvar)
        if best is None:
            raise InfeasibleAssignment("no receding-horizon action satisfies all hard constraints")
        _, proposed, acceptance, regrets, objectives, safety_cvar = best
        return RecedingHorizonPlan(
            first_assignments=proposed,
            acceptance_probabilities=acceptance,
            dynamic_regrets=regrets,
            horizon_objectives=tuple(objectives),
            expected_safety_cvar=safety_cvar,
            baseline_safety_cvar=baseline_cvar,
            objective=sum((self.discount**i) * value for i, value in enumerate(objectives)),
            solve_time_seconds=time.perf_counter() - started,
            combinations_evaluated=evaluated,
        )
