from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from concordia.models import Route, User
from concordia.network import RoadNetwork
from concordia.preferences import UtilityModel, preference_slack


@dataclass(frozen=True)
class AlignmentPotential:
    score: float
    opportunity_mass: float
    opportunity_count: int
    maximum_single_benefit: float
    slack_mass: float


def _flows(
    network: RoadNetwork,
    routes: Mapping[str, Route],
    assignments: Mapping[str, str],
    vehicle_flow: float,
) -> dict:
    flows = {edge: 0.0 for edge in network.edges}
    for route_id in assignments.values():
        for edge in routes[route_id].edges:
            flows[edge] += vehicle_flow
    return flows


def _ttt(network: RoadNetwork, flows: Mapping[tuple[str, str], float]) -> float:
    return sum(
        flow * network.edge_data(edge).travel_time(flow) for edge, flow in flows.items()
    )


def compute_alignment_potential(
    network: RoadNetwork,
    routes: Mapping[str, Route],
    users: Sequence[User],
    candidates: Mapping[str, Sequence[str]],
    vehicle_flow: float,
    demand: float,
) -> AlignmentPotential:
    """Project-defined APS/AOM from individually feasible one-user route moves."""
    utility_model = UtilityModel()
    slacks = {}
    private = {}
    for user in users:
        utilities = utility_model.utilities(
            user.preferences, (routes[route_id] for route_id in candidates[user.user_id])
        )
        slacks[user.user_id] = preference_slack(utilities)
        private[user.user_id] = min(
            sorted(candidates[user.user_id]), key=slacks[user.user_id].__getitem__
        )
    baseline_ttt = _ttt(network, _flows(network, routes, private, vehicle_flow))
    benefits = []
    feasible_slacks = []
    for user in users:
        for route_id in candidates[user.user_id]:
            if route_id == private[user.user_id]:
                continue
            slack = float(slacks[user.user_id][route_id])
            if slack > user.epsilon + 1e-10:
                continue
            trial = dict(private)
            trial[user.user_id] = route_id
            marginal_network_benefit = max(
                0.0,
                baseline_ttt - _ttt(network, _flows(network, routes, trial, vehicle_flow)),
            )
            if marginal_network_benefit > 0:
                benefits.append(marginal_network_benefit)
                feasible_slacks.append(slack)
    mass = float(sum(benefits))
    return AlignmentPotential(
        score=mass / max(float(demand), 1e-12),
        opportunity_mass=mass,
        opportunity_count=len(benefits),
        maximum_single_benefit=max(benefits, default=0.0),
        slack_mass=float(sum(feasible_slacks)),
    )
