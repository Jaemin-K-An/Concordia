from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping, Optional

from concordia.behavior import RecommendationDecision
from concordia.errors import ValidationError


@dataclass(frozen=True)
class EdgeObservation:
    """Instantaneous edge quantities with explicit units.

    Flow is the traffic-state estimate q=k*v in vehicles/hour/lane, not detector throughput.
    """

    vehicle_count: int
    density_vehicles_per_km_per_lane: float
    flow_vehicles_per_hour_per_lane: float
    mean_speed_meters_per_second: float
    occupancy_percent: Optional[float]
    lane_count: int
    length_meters: float
    speed_coefficient_of_variation: float = 0.0
    acceleration_variance_meters2_per_second4: float = 0.0
    headway_mean_seconds: Optional[float] = None
    headway_variance_seconds2: Optional[float] = None

    def __post_init__(self) -> None:
        if self.vehicle_count < 0 or self.lane_count < 1 or self.length_meters <= 0:
            raise ValidationError("edge count/geometry quantities are invalid")
        values = (
            self.density_vehicles_per_km_per_lane,
            self.flow_vehicles_per_hour_per_lane,
            self.mean_speed_meters_per_second,
            self.speed_coefficient_of_variation,
            self.acceleration_variance_meters2_per_second4,
        )
        if any(value < 0 for value in values):
            raise ValidationError("edge traffic quantities cannot be negative")
        if self.occupancy_percent is not None and not 0 <= self.occupancy_percent <= 100:
            raise ValidationError("edge occupancy must be a percentage")
        headways = (self.headway_mean_seconds, self.headway_variance_seconds2)
        if any(value is not None and value < 0 for value in headways):
            raise ValidationError("edge headway statistics cannot be negative")


@dataclass(frozen=True)
class SimulationSnapshot:
    time: float
    edges: Mapping[str, EdgeObservation]

    def __post_init__(self) -> None:
        if self.time < 0:
            raise ValidationError("simulation time cannot be negative")


class SimulationAdapter(ABC):
    """Boundary that keeps recommendation logic independent of simulator APIs."""

    @abstractmethod
    def start(self, seed: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def step(self) -> SimulationSnapshot:
        raise NotImplementedError

    @abstractmethod
    def execute_accepted_route(self, decision: RecommendationDecision) -> bool:
        """Apply a route only after a domain-level accepted decision; return whether applied."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError
