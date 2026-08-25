from __future__ import annotations

import itertools
import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from concordia.errors import InfeasibleAssignment, ValidationError
from concordia.models import AssignmentResult, EdgeKey, Route, User
from concordia.network import RoadNetwork
from concordia.preferences import UtilityModel, preference_slack
from concordia.traffic import GhostRiskModel


@dataclass(frozen=True)
class ObjectiveWeights:
    ghost_risk: float = 0.0
    safety_risk: float = 0.0
    concentration: float = 0.0

    def __post_init__(self) -> None:
        if min(self.ghost_risk, self.safety_risk, self.concentration) < 0:
            raise ValidationError("objective weights cannot be negative")


class AdaptiveOptimizer:
    """Exact oracle and explicit greedy baseline for route recommendations."""

    def __init__(
        self,
        network: RoadNetwork,
        routes: Mapping[str, Route],
        utility_model: Optional[UtilityModel] = None,
        objective_weights: Optional[ObjectiveWeights] = None,
        vehicle_flow: float = 1.0,
    ) -> None:
        if not routes or vehicle_flow <= 0:
            raise ValidationError("routes and positive vehicle_flow are required")
        self.network = network
        self.routes = dict(routes)
        self.utility_model = utility_model or UtilityModel()
        self.weights = objective_weights or ObjectiveWeights()
        self.vehicle_flow = vehicle_flow
        self.ghost_model = GhostRiskModel()
        for route in self.routes.values():
            self.network.path_edges(route.nodes)

    def _user_slack(self, user: User, route_ids: Iterable[str]) -> Dict[str, float]:
        candidates = [self.routes[route_id] for route_id in route_ids]
        utilities = self.utility_model.utilities(user.preferences, candidates)
        return preference_slack(utilities)

    def _flows(self, assignments: Mapping[str, str]) -> Dict[EdgeKey, float]:
        flows = {edge: 0.0 for edge in self.network.edges}
        for route_id in assignments.values():
            for edge in self.routes[route_id].edges:
                flows[edge] += self.vehicle_flow
        return flows

    def _metrics(
        self,
        assignments: Mapping[str, str],
        regrets: Mapping[str, float],
    ) -> AssignmentResult:
        flows = self._flows(assignments)
        total_travel_time = sum(
            flow * self.network.edge_data(edge).travel_time(flow) for edge, flow in flows.items()
        )
        total_safety = sum(
            self.routes[route_id].features.risk * self.vehicle_flow
            for route_id in assignments.values()
        )
        total_ghost = sum(
            self.ghost_model.probability(
                saturation=flow / self.network.edge_data(edge).capacity,
                speed_cv=0.0,
                acceleration_variance=0.0,
            )
            * flow
            for edge, flow in flows.items()
            if flow > 0
        )
        counts = Counter(assignments.values())
        total = len(assignments)
        shares = [count / total for count in counts.values()]
        entropy = -sum(share * math.log(share) for share in shares if share > 0)
        concentration = sum(share**2 for share in shares)
        objective = (
            total_travel_time
            + self.weights.ghost_risk * total_ghost
            + self.weights.safety_risk * total_safety
            + self.weights.concentration * concentration
        )
        return AssignmentResult(
            assignments=dict(assignments),
            objective=objective,
            total_travel_time=total_travel_time,
            total_safety_risk=total_safety,
            total_ghost_risk=total_ghost,
            route_entropy=entropy,
            regrets=dict(regrets),
            metadata={"route_concentration_hhi": concentration, "flows": flows},
        )

    def private_best(
        self, users: Sequence[User], candidates: Mapping[str, Sequence[str]]
    ) -> AssignmentResult:
        assignments: Dict[str, str] = {}
        regrets: Dict[str, float] = {}
        for user in users:
            slacks = self._user_slack(user, candidates[user.user_id])
            assignments[user.user_id] = min(sorted(slacks), key=slacks.__getitem__)
            regrets[user.user_id] = 0.0
        return self._metrics(assignments, regrets)

    def evaluate(
        self,
        assignments: Mapping[str, str],
        regrets: Optional[Mapping[str, float]] = None,
    ) -> AssignmentResult:
        """Evaluate a fully specified assignment using the same objective implementation."""
        if not assignments:
            raise ValidationError("cannot evaluate an empty assignment")
        unknown = set(assignments.values()) - set(self.routes)
        if unknown:
            raise ValidationError(f"assignment references unknown routes: {sorted(unknown)}")
        supplied_regrets = dict(regrets or {user_id: 0.0 for user_id in assignments})
        if set(supplied_regrets) != set(assignments) or any(value < 0 for value in supplied_regrets.values()):
            raise ValidationError("regret keys must match assignments and values cannot be negative")
        return self._metrics(assignments, supplied_regrets)

    def exact(
        self,
        users: Sequence[User],
        candidates: Mapping[str, Sequence[str]],
        safety_delta: float = 0.0,
        max_combinations: int = 1_000_000,
    ) -> AssignmentResult:
        if not users or safety_delta < 0:
            raise ValidationError("users are required and safety_delta cannot be negative")
        admissible: List[List[Tuple[str, float]]] = []
        combinations = 1
        for user in users:
            if user.user_id not in candidates or not candidates[user.user_id]:
                raise ValidationError(f"missing candidates for {user.user_id}")
            slacks = self._user_slack(user, candidates[user.user_id])
            options = [
                (route_id, slack)
                for route_id, slack in slacks.items()
                if slack <= user.epsilon + 1e-10
            ]
            if not options:
                raise InfeasibleAssignment(f"no preference-compatible route for {user.user_id}")
            combinations *= len(options)
            if combinations > max_combinations:
                raise ValidationError(
                    f"exact oracle would enumerate {combinations} combinations; use greedy explicitly"
                )
            admissible.append(options)
        baseline = self.private_best(users, candidates)
        safety_limit = baseline.total_safety_risk + safety_delta
        best: Optional[AssignmentResult] = None
        for combination in itertools.product(*admissible):
            assignments = {
                user.user_id: option[0] for user, option in zip(users, combination)
            }
            regrets = {user.user_id: option[1] for user, option in zip(users, combination)}
            result = self._metrics(assignments, regrets)
            if result.total_safety_risk > safety_limit + 1e-10:
                continue
            if best is None or (result.objective, sorted(result.assignments.items())) < (
                best.objective,
                sorted(best.assignments.items()),
            ):
                best = result
        if best is None:
            raise InfeasibleAssignment("no assignment satisfies the aggregate safety constraint")
        return AssignmentResult(
            **{
                **best.__dict__,
                "metadata": {
                    **best.metadata,
                    "method": "exact_enumeration",
                    "combinations": combinations,
                    "safety_limit": safety_limit,
                },
            }
        )

    def greedy_vde(
        self,
        users: Sequence[User],
        candidates: Mapping[str, Sequence[str]],
        safety_delta: float = 0.0,
    ) -> AssignmentResult:
        baseline = self.private_best(users, candidates)
        assignments = dict(baseline.assignments)
        regrets = {user.user_id: 0.0 for user in users}
        safety_limit = baseline.total_safety_risk + safety_delta
        current = baseline
        while True:
            choices = []
            for user in users:
                slacks = self._user_slack(user, candidates[user.user_id])
                for route_id, slack in slacks.items():
                    if route_id == assignments[user.user_id] or slack > user.epsilon + 1e-10:
                        continue
                    trial_assignments = {**assignments, user.user_id: route_id}
                    trial_regrets = {**regrets, user.user_id: slack}
                    trial = self._metrics(trial_assignments, trial_regrets)
                    benefit = current.objective - trial.objective
                    vde = benefit / (slack + 1e-9)
                    if benefit > 1e-12 and trial.total_safety_risk <= safety_limit + 1e-10:
                        choices.append((vde, benefit, user.user_id, route_id, slack, trial))
            if not choices:
                break
            _, _, user_id, route_id, slack, current = max(
                choices, key=lambda item: (item[0], item[1], item[2], item[3])
            )
            assignments[user_id] = route_id
            regrets[user_id] = slack
        return AssignmentResult(
            **{
                **current.__dict__,
                "metadata": {
                    **current.metadata,
                    "method": "greedy_vde",
                    "safety_limit": safety_limit,
                },
            }
        )
