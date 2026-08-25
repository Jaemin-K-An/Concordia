from __future__ import annotations

from collections import defaultdict
from typing import Dict, Mapping, Sequence

from concordia.errors import ValidationError
from concordia.models import AssignmentResult, User
from concordia.optimization.adaptive import AdaptiveOptimizer
from concordia.preferences import UtilityModel, preference_slack


def clustered_greedy_assignment(
    optimizer: AdaptiveOptimizer,
    users: Sequence[User],
    candidates: Mapping[str, Sequence[str]],
    *,
    maximum_clusters: int = 64,
    safety_delta: float = 0.0,
) -> AssignmentResult:
    """Reduced-candidate deterministic approximation with hard regret/safety filters.

    Users are grouped by their dominant preference dimension and private-best route. A
    cluster move is allowed only for members whose individual slack respects epsilon.
    """
    if not users or maximum_clusters < 1 or safety_delta < 0:
        raise ValidationError("clustered approximation parameters are invalid")
    utility_model = UtilityModel()
    slacks = {}
    private = {}
    clusters = defaultdict(list)
    for user in users:
        route_ids = tuple(candidates.get(user.user_id, ()))
        if not route_ids:
            raise ValidationError(f"missing candidates for {user.user_id}")
        utilities = utility_model.utilities(
            user.preferences, (optimizer.routes[route_id] for route_id in route_ids)
        )
        slacks[user.user_id] = preference_slack(utilities)
        private_route = min(sorted(route_ids), key=slacks[user.user_id].__getitem__)
        private[user.user_id] = private_route
        dominant = max(
            user.preferences.normalized().as_dict(),
            key=user.preferences.normalized().as_dict().__getitem__,
        )
        clusters[(dominant, private_route)].append(user)
    if len(clusters) > maximum_clusters:
        ordered = sorted(clusters.items(), key=lambda item: (-len(item[1]), item[0]))
        retained = dict(ordered[: maximum_clusters - 1])
        overflow = [user for _, members in ordered[maximum_clusters - 1 :] for user in members]
        retained[("overflow", "mixed")] = overflow
        clusters = defaultdict(list, retained)

    assignments: Dict[str, str] = dict(private)
    regrets = {user.user_id: 0.0 for user in users}
    baseline = optimizer.evaluate(assignments, regrets)
    safety_limit = baseline.total_safety_risk + safety_delta
    current = baseline
    moves = 0
    for _, members in sorted(clusters.items(), key=lambda item: (-len(item[1]), item[0])):
        route_options = sorted(
            {route_id for user in members for route_id in candidates[user.user_id]}
        )
        best = None
        for route_id in route_options:
            eligible = [
                user
                for user in members
                if route_id in candidates[user.user_id]
                and slacks[user.user_id][route_id] <= user.epsilon + 1e-10
            ]
            if not eligible:
                continue
            trial_assignments = dict(assignments)
            trial_regrets = dict(regrets)
            for user in eligible:
                trial_assignments[user.user_id] = route_id
                trial_regrets[user.user_id] = slacks[user.user_id][route_id]
            trial = optimizer.evaluate(trial_assignments, trial_regrets)
            if trial.total_safety_risk > safety_limit + 1e-10:
                continue
            key = (trial.objective, route_id)
            if best is None or key < best[0]:
                best = (key, trial, trial_assignments, trial_regrets)
        if best is not None and best[1].objective < current.objective - 1e-10:
            _, current, assignments, regrets = best
            moves += 1
    return AssignmentResult(
        **{
            **current.__dict__,
            "metadata": {
                **current.metadata,
                "method": "dominant_preference_clustered_greedy",
                "cluster_count": len(clusters),
                "cluster_moves": moves,
                "safety_limit": safety_limit,
            },
        }
    )
