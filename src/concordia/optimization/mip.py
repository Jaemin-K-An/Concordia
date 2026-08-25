from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass
from importlib import metadata
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from concordia.errors import InfeasibleAssignment, SolverUnavailable, ValidationError
from concordia.models import Route, User
from concordia.network import RoadNetwork
from concordia.preferences import preference_slack


@dataclass(frozen=True)
class MIPAssignmentResult:
    assignments: Mapping[str, str]
    objective: float
    total_safety_risk: float
    regrets: Mapping[str, float]
    solve_time_seconds: float
    solver: str
    optimal: bool


class MIPAssignmentSolver:
    """Open-source HiGHS binary assignment over linearized route externality costs.

    This scalable baseline does not pretend the nonlinear BPR objective is linear. Callers must
    provide costs linearized at the current network state; exact enumeration remains the small
    nonlinear correctness oracle.
    """

    def __init__(
        self,
        network: RoadNetwork,
        routes: Mapping[str, Route],
        minimum_acceptance_probability: float = 0.0,
        time_limit_seconds: float = 10.0,
    ) -> None:
        if not 0 <= minimum_acceptance_probability <= 1 or time_limit_seconds <= 0:
            raise ValidationError("MIP acceptance threshold/time limit is invalid")
        self.network = network
        self.routes = dict(routes)
        self.minimum_acceptance_probability = minimum_acceptance_probability
        self.time_limit_seconds = time_limit_seconds
        for route in self.routes.values():
            self.network.path_edges(route.nodes)

    @staticmethod
    def available() -> bool:
        return importlib.util.find_spec("scipy.optimize") is not None

    def solve(
        self,
        users: Sequence[User],
        candidates: Mapping[str, Sequence[str]],
        utilities: Mapping[str, Mapping[str, float]],
        linearized_costs: Mapping[Tuple[str, str], float],
        acceptance_probabilities: Mapping[Tuple[str, str], float],
        baseline_assignments: Mapping[str, str],
        safety_delta: float = 0.0,
    ) -> MIPAssignmentResult:
        if not self.available():
            raise SolverUnavailable("SciPy/HiGHS is required for the requested MIP baseline")
        if not users or safety_delta < 0:
            raise ValidationError("MIP users/safety delta are invalid")
        from scipy.optimize import Bounds, LinearConstraint, milp

        variables = []
        regret_by_variable = []
        upper_bounds = []
        costs = []
        risks = []
        for user in users:
            user_id = user.user_id
            route_ids = tuple(candidates.get(user_id, ()))
            if not route_ids or user_id not in utilities:
                raise ValidationError(f"missing MIP inputs for {user_id}")
            slacks = preference_slack(utilities[user_id])
            for route_id in route_ids:
                key = (user_id, route_id)
                if route_id not in self.routes or key not in linearized_costs:
                    raise ValidationError(f"missing MIP route/cost for {key}")
                probability = float(acceptance_probabilities.get(key, 0.0))
                admissible = (
                    slacks[route_id] <= user.epsilon + 1e-10
                    and probability >= self.minimum_acceptance_probability
                )
                variables.append(key)
                regret_by_variable.append(slacks[route_id])
                upper_bounds.append(1.0 if admissible else 0.0)
                costs.append(float(linearized_costs[key]))
                risks.append(self.routes[route_id].features.risk)
        baseline_risk = sum(
            self.routes[baseline_assignments[user.user_id]].features.risk for user in users
        )
        safety_limit = baseline_risk + safety_delta
        rows = []
        lower = []
        upper = []
        for user in users:
            rows.append([1.0 if key[0] == user.user_id else 0.0 for key in variables])
            lower.append(1.0)
            upper.append(1.0)
        rows.append(risks)
        lower.append(-np.inf)
        upper.append(safety_limit)
        started = time.perf_counter()
        result = milp(
            c=np.asarray(costs, dtype=float),
            integrality=np.ones(len(variables), dtype=int),
            bounds=Bounds(np.zeros(len(variables)), np.asarray(upper_bounds)),
            constraints=LinearConstraint(np.asarray(rows), np.asarray(lower), np.asarray(upper)),
            options={"time_limit": self.time_limit_seconds},
        )
        elapsed = time.perf_counter() - started
        if not result.success or result.x is None:
            if result.status == 1:
                raise SolverUnavailable(f"MIP time/iteration limit reached: {result.message}")
            raise InfeasibleAssignment(f"MIP failed: {result.message}")
        assignments: Dict[str, str] = {}
        regrets: Dict[str, float] = {}
        total_risk = 0.0
        for index, value in enumerate(result.x):
            if value > 0.5:
                user_id, route_id = variables[index]
                assignments[user_id] = route_id
                regrets[user_id] = regret_by_variable[index]
                total_risk += risks[index]
        if set(assignments) != {user.user_id for user in users}:
            raise InfeasibleAssignment("MIP solution does not assign every user exactly once")
        try:
            scipy_version = metadata.version("scipy")
        except metadata.PackageNotFoundError:
            scipy_version = "unknown"
        return MIPAssignmentResult(
            assignments=assignments,
            objective=float(result.fun),
            total_safety_risk=total_risk,
            regrets=regrets,
            solve_time_seconds=elapsed,
            solver=f"scipy-{scipy_version}/HiGHS",
            optimal=result.status == 0,
        )

