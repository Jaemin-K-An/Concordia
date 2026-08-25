from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

from concordia.errors import SimulatorUnavailable, ValidationError
from concordia.simulation.base import SimulationAdapter, SimulationSnapshot


class SumoAdapter(SimulationAdapter):
    def __init__(self, config_path: str, binary: str = "sumo") -> None:
        self.config_path = Path(config_path)
        self.binary = binary
        self._traci = None

    def _validate_environment(self) -> str:
        executable = shutil.which(self.binary)
        if executable is None or importlib.util.find_spec("traci") is None:
            raise SimulatorUnavailable(
                "SUMO/TraCI was requested but is unavailable; install SUMO and the 'sumo' extra"
            )
        if not self.config_path.is_file():
            raise ValidationError(f"SUMO config does not exist: {self.config_path}")
        return executable

    def start(self, seed: int) -> None:
        if seed < 0:
            raise ValidationError("SUMO seed must be non-negative")
        executable = self._validate_environment()
        import traci  # type: ignore

        traci.start([executable, "-c", str(self.config_path), "--seed", str(seed)])
        self._traci = traci

    def step(self) -> SimulationSnapshot:
        if self._traci is None:
            raise ValidationError("SUMO adapter has not been started")
        self._traci.simulationStep()
        edge_ids = tuple(self._traci.edge.getIDList())
        return SimulationSnapshot(
            time=float(self._traci.simulation.getTime()),
            edge_speed={edge: float(self._traci.edge.getLastStepMeanSpeed(edge)) for edge in edge_ids},
            edge_density={
                edge: float(self._traci.edge.getLastStepVehicleNumber(edge)) for edge in edge_ids
            },
            edge_flow={
                edge: float(self._traci.edge.getLastStepVehicleNumber(edge)) for edge in edge_ids
            },
        )

    def recommend_route(self, vehicle_id: str, edge_ids: list[str]) -> None:
        if self._traci is None:
            raise ValidationError("SUMO adapter has not been started")
        if not vehicle_id or not edge_ids:
            raise ValidationError("vehicle id and a non-empty route are required")
        self._traci.vehicle.setRoute(vehicle_id, edge_ids)

    def close(self) -> None:
        if self._traci is not None:
            self._traci.close()
            self._traci = None
