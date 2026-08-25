from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

from concordia.errors import ValidationError
from concordia.models import EdgeKey
from concordia.network import RoadNetwork
from concordia.simulation import SimulationSnapshot
from concordia.traffic import GhostRiskModel


@dataclass(frozen=True)
class EdgeState:
    flow_vehicles_per_hour: float
    density_vehicles_per_km_per_lane: float
    mean_speed_meters_per_second: float
    vehicle_count: int
    occupancy_percent: Optional[float]
    saturation: float
    ghost_risk_probability: float
    safety_risk_index: float

    def __post_init__(self) -> None:
        values = (
            self.flow_vehicles_per_hour,
            self.density_vehicles_per_km_per_lane,
            self.mean_speed_meters_per_second,
            self.saturation,
            self.ghost_risk_probability,
            self.safety_risk_index,
        )
        if self.vehicle_count < 0 or any(value < 0 for value in values):
            raise ValidationError("network edge state cannot contain negative quantities")
        if self.ghost_risk_probability > 1:
            raise ValidationError("ghost risk must be a probability")


@dataclass(frozen=True)
class NetworkState:
    timestamp_seconds: float
    edges: Mapping[EdgeKey, EdgeState]
    source: str

    def __post_init__(self) -> None:
        if self.timestamp_seconds < 0 or not self.edges or not self.source:
            raise ValidationError("network state requires time, edges, and provenance")

    @property
    def flows(self) -> Dict[EdgeKey, float]:
        return {edge: state.flow_vehicles_per_hour for edge, state in self.edges.items()}


class NetworkStateEstimator:
    def __init__(self, network: RoadNetwork, ghost_model: Optional[GhostRiskModel] = None) -> None:
        self.network = network
        self.ghost_model = ghost_model or GhostRiskModel()

    def from_snapshot(
        self,
        snapshot: SimulationSnapshot,
        edge_id_map: Mapping[str, EdgeKey],
        source: str,
    ) -> NetworkState:
        if set(snapshot.edges) - set(edge_id_map):
            raise ValidationError("snapshot contains edge ids without physical-graph mapping")
        states = {}
        for edge_id, observation in snapshot.edges.items():
            edge = edge_id_map[edge_id]
            data = self.network.edge_data(edge)
            total_flow = observation.flow_vehicles_per_hour_per_lane * observation.lane_count
            saturation = total_flow / data.capacity
            states[edge] = EdgeState(
                flow_vehicles_per_hour=total_flow,
                density_vehicles_per_km_per_lane=observation.density_vehicles_per_km_per_lane,
                mean_speed_meters_per_second=observation.mean_speed_meters_per_second,
                vehicle_count=observation.vehicle_count,
                occupancy_percent=observation.occupancy_percent,
                saturation=saturation,
                ghost_risk_probability=self.ghost_model.probability(saturation, 0.0, 0.0),
                safety_risk_index=data.risk,
            )
        missing = set(self.network.edges) - set(states)
        if missing:
            raise ValidationError(f"snapshot is incomplete for physical edges: {sorted(missing)}")
        return NetworkState(snapshot.time, states, source)

    def from_flows(
        self,
        flows: Mapping[EdgeKey, float],
        timestamp_seconds: float,
        source: str = "analytical_bpr",
    ) -> NetworkState:
        states = {}
        for edge in self.network.edges:
            data = self.network.edge_data(edge)
            flow = float(flows.get(edge, 0.0))
            if flow < 0:
                raise ValidationError("analytical state flow cannot be negative")
            travel_time_hours = data.travel_time(flow) / 60.0
            speed_kph = data.length / travel_time_hours if travel_time_hours > 0 else 0.0
            density = flow / speed_kph if speed_kph > 0 else 0.0
            saturation = flow / data.capacity
            states[edge] = EdgeState(
                flow_vehicles_per_hour=flow,
                density_vehicles_per_km_per_lane=density,
                mean_speed_meters_per_second=speed_kph / 3.6,
                vehicle_count=max(0, round(density * data.length)),
                occupancy_percent=None,
                saturation=saturation,
                ghost_risk_probability=self.ghost_model.probability(saturation, 0.0, 0.0),
                safety_risk_index=data.risk,
            )
        return NetworkState(timestamp_seconds, states, source)

