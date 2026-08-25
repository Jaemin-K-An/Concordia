from __future__ import annotations

from typing import Dict, Mapping

from concordia.behavior import RecommendationDecision
from concordia.errors import ValidationError
from concordia.models import EdgeKey, Route
from concordia.network import RoadNetwork
from concordia.simulation.base import EdgeObservation, SimulationAdapter, SimulationSnapshot


def analytical_edge_id(edge: EdgeKey) -> str:
    return f"{edge[0]}->{edge[1]}"


class AnalyticalSimulationAdapter(SimulationAdapter):
    """Deterministic BPR state adapter for closed-loop tests; not microscopic simulation."""

    def __init__(
        self,
        network: RoadNetwork,
        routes: Mapping[str, Route],
        assignments: Mapping[str, str],
        vehicle_flow: float,
        step_seconds: float = 30.0,
    ) -> None:
        if not assignments or vehicle_flow <= 0 or step_seconds <= 0:
            raise ValidationError("analytical simulator requires assignments and positive units")
        self.network = network
        self.routes = dict(routes)
        self.assignments = dict(assignments)
        self.vehicle_flow = vehicle_flow
        self.step_seconds = step_seconds
        self.time = 0.0
        self.seed = None

    @property
    def edge_id_map(self) -> Dict[str, EdgeKey]:
        return {analytical_edge_id(edge): edge for edge in self.network.edges}

    def start(self, seed: int) -> None:
        if seed < 0:
            raise ValidationError("analytical seed must be non-negative")
        self.seed = seed
        self.time = 0.0

    def _flows(self) -> Dict[EdgeKey, float]:
        flows = {edge: 0.0 for edge in self.network.edges}
        for route_id in self.assignments.values():
            if route_id not in self.routes:
                raise ValidationError(f"unknown analytical route assignment: {route_id}")
            for edge in self.routes[route_id].edges:
                flows[edge] += self.vehicle_flow
        return flows

    def step(self) -> SimulationSnapshot:
        if self.seed is None:
            raise ValidationError("analytical simulator has not been started")
        self.time += self.step_seconds
        observations = {}
        for edge, flow in self._flows().items():
            data = self.network.edge_data(edge)
            travel_time_hours = data.travel_time(flow) / 60.0
            speed_kph = data.length / travel_time_hours
            density = flow / speed_kph if speed_kph > 0 else 0.0
            observations[analytical_edge_id(edge)] = EdgeObservation(
                vehicle_count=max(0, round(density * data.length)),
                density_vehicles_per_km_per_lane=density,
                flow_vehicles_per_hour_per_lane=flow,
                mean_speed_meters_per_second=speed_kph / 3.6,
                occupancy_percent=None,
                lane_count=1,
                length_meters=data.length * 1000.0,
            )
        return SimulationSnapshot(self.time, observations)

    def execute_accepted_route(self, decision: RecommendationDecision) -> bool:
        if not decision.accepted:
            return False
        route_id = decision.offer.candidate_route_id
        if route_id not in self.routes or decision.offer.user_id not in self.assignments:
            raise ValidationError("accepted analytical decision references an unknown user/route")
        self.assignments[decision.offer.user_id] = route_id
        return True

    def close(self) -> None:
        self.seed = None

