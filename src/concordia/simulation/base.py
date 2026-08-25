from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class SimulationSnapshot:
    time: float
    edge_speed: Mapping[str, float]
    edge_density: Mapping[str, float]
    edge_flow: Mapping[str, float]


class SimulationAdapter(ABC):
    """Boundary that keeps recommendation logic independent of simulator APIs."""

    @abstractmethod
    def start(self, seed: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def step(self) -> SimulationSnapshot:
        raise NotImplementedError

    @abstractmethod
    def recommend_route(self, vehicle_id: str, edge_ids: list[str]) -> None:
        """Change only the route recommendation/assignment, never vehicle controls."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError
