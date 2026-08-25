from __future__ import annotations

import importlib.util
import shutil
import statistics
from pathlib import Path
from typing import Optional

from concordia.behavior import RecommendationDecision
from concordia.errors import SimulatorUnavailable, ValidationError
from concordia.simulation.base import EdgeObservation, SimulationAdapter, SimulationSnapshot


class SumoAdapter(SimulationAdapter):
    def __init__(self, config_path: str, binary: str = "sumo") -> None:
        self.config_path = Path(config_path)
        self.binary = binary
        self._traci = None

    @staticmethod
    def resolve_binary(binary: str) -> Optional[str]:
        executable = shutil.which(binary)
        if executable is not None:
            return executable
        if importlib.util.find_spec("sumo") is not None:
            import sumo  # type: ignore

            packaged = Path(sumo.SUMO_HOME) / "bin" / binary
            if packaged.is_file():
                return str(packaged)
        return None

    def _validate_environment(self) -> str:
        executable = self.resolve_binary(self.binary)
        if executable is None or importlib.util.find_spec("traci") is None:
            raise SimulatorUnavailable(
                "SUMO/TraCI was requested but is unavailable; install SUMO and the 'sumo' extra"
            )
        if not self.config_path.is_file():
            raise ValidationError(f"SUMO config does not exist: {self.config_path}")
        return executable

    @classmethod
    def simulator_version(cls, binary: str = "sumo") -> str:
        executable = cls.resolve_binary(binary)
        if executable is None:
            raise SimulatorUnavailable(f"SUMO binary is unavailable: {binary}")
        import subprocess

        output = subprocess.check_output([executable, "--version"], text=True)
        return output.splitlines()[0].strip()

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
        observations = {}
        for edge_id in self._traci.edge.getIDList():
            vehicle_count = int(self._traci.edge.getLastStepVehicleNumber(edge_id))
            lane_count = int(self._traci.edge.getLaneNumber(edge_id))
            length_meters = float(self._traci.lane.getLength(f"{edge_id}_0"))
            mean_speed_mps = max(0.0, float(self._traci.edge.getLastStepMeanSpeed(edge_id)))
            lane_kilometers = lane_count * length_meters / 1000.0
            density = vehicle_count / lane_kilometers
            flow_estimate = density * mean_speed_mps * 3.6
            occupancy = float(self._traci.edge.getLastStepOccupancy(edge_id))
            vehicle_id_getter = getattr(self._traci.edge, "getLastStepVehicleIDs", None)
            vehicle_ids = (
                tuple(vehicle_id_getter(edge_id)) if callable(vehicle_id_getter) else ()
            )
            speeds = [max(0.0, float(self._traci.vehicle.getSpeed(item))) for item in vehicle_ids]
            accelerations = [
                float(self._traci.vehicle.getAcceleration(item)) for item in vehicle_ids
            ]
            speed_cv = (
                statistics.pstdev(speeds) / statistics.fmean(speeds)
                if len(speeds) > 1 and statistics.fmean(speeds) > 1e-12
                else 0.0
            )
            acceleration_variance = (
                statistics.pvariance(accelerations) if len(accelerations) > 1 else 0.0
            )
            headways = []
            for vehicle_id in vehicle_ids:
                leader = self._traci.vehicle.getLeader(vehicle_id, 200.0)
                vehicle_speed = max(0.0, float(self._traci.vehicle.getSpeed(vehicle_id)))
                if leader and vehicle_speed > 1e-6:
                    headways.append(max(0.0, float(leader[1])) / vehicle_speed)
            observations[edge_id] = EdgeObservation(
                vehicle_count=vehicle_count,
                density_vehicles_per_km_per_lane=density,
                flow_vehicles_per_hour_per_lane=flow_estimate,
                mean_speed_meters_per_second=mean_speed_mps,
                occupancy_percent=occupancy,
                lane_count=lane_count,
                length_meters=length_meters,
                speed_coefficient_of_variation=speed_cv,
                acceleration_variance_meters2_per_second4=acceleration_variance,
                headway_mean_seconds=(statistics.fmean(headways) if headways else None),
                headway_variance_seconds2=(
                    statistics.pvariance(headways) if len(headways) > 1 else None
                ),
            )
        return SimulationSnapshot(time=float(self._traci.simulation.getTime()), edges=observations)

    def execute_accepted_route(self, decision: RecommendationDecision) -> bool:
        if self._traci is None:
            raise ValidationError("SUMO adapter has not been started")
        if not decision.accepted:
            return False
        self._traci.vehicle.setRoute(
            decision.offer.user_id,
            list(decision.offer.executable_edge_ids),
        )
        return True

    def close(self) -> None:
        if self._traci is not None:
            self._traci.close()
            self._traci = None
