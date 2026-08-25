#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from concordia.safety import (
    TrajectoryFrame,
    parse_sumo_ssm,
    summarize_safety,
    summarize_ssm_conflict_types,
)
from concordia.evaluation import ExperimentRegistry, capture_source_state
from concordia.simulation import SumoAdapter
from concordia.traffic import DetectorObservation, PhantomJamEventDetector


ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "scenarios" / "sumo" / "ring"
ARTIFACT = ROOT / "artifacts" / "studies" / "sumo_ring"


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def distribution_summary(values: list[float], risk_tail: str) -> dict:
    if risk_tail not in {"lower", "upper"}:
        raise ValueError("risk_tail must be lower or upper")
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "minimum": None,
            "maximum": None,
            "risk_tail": risk_tail,
            "risk_cvar": None,
            "cvar05_lower": None,
            "cvar95_upper": None,
        }
    data = np.asarray(values, dtype=float)
    lower_threshold = float(np.percentile(data, 5))
    upper_threshold = float(np.percentile(data, 95))
    lower_cvar = float(data[data <= lower_threshold].mean())
    upper_cvar = float(data[data >= upper_threshold].mean())
    return {
        "count": int(len(data)),
        "mean": float(data.mean()),
        "median": float(np.median(data)),
        "p90": float(np.percentile(data, 90)),
        "p95": upper_threshold,
        "p99": float(np.percentile(data, 99)),
        "minimum": float(data.min()),
        "maximum": float(data.max()),
        "risk_tail": risk_tail,
        "risk_cvar": lower_cvar if risk_tail == "lower" else upper_cvar,
        "cvar05_lower": lower_cvar,
        "cvar95_upper": upper_cvar,
    }


def build_network() -> Path:
    netconvert = SumoAdapter.resolve_binary("netconvert")
    if netconvert is None:
        raise SystemExit("netconvert is unavailable")
    output = SCENARIO / "ring.net.xml"
    subprocess.run(
        [
            netconvert,
            "--node-files",
            str(SCENARIO / "ring.nod.xml"),
            "--edge-files",
            str(SCENARIO / "ring.edg.xml"),
            "--output-file",
            str(output),
            "--no-turnarounds",
            "true",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    if not output.is_file():
        raise SystemExit("netconvert completed without ring.net.xml")
    return output


def run() -> Path:
    source_commit, source_dirty = capture_source_state()
    started = datetime.now(timezone.utc)
    build_network()
    binary = SumoAdapter.resolve_binary("sumo")
    if binary is None:
        raise SystemExit("sumo is unavailable")
    adapter = SumoAdapter(str(SCENARIO / "ring.sumocfg"), binary=binary)
    snapshots = []
    frames = []
    vehicle_time_loss = {}
    arrived_vehicles = 0
    departed_vehicles = 0
    maximum_halting_vehicles = 0
    adapter.start(seed=42)
    try:
        while adapter._traci.simulation.getMinExpectedNumber() > 0:
            snapshot = adapter.step()
            snapshots.append(snapshot)
            arrived_vehicles += int(adapter._traci.simulation.getArrivedNumber())
            departed_vehicles += int(adapter._traci.simulation.getDepartedNumber())
            maximum_halting_vehicles = max(
                maximum_halting_vehicles,
                sum(
                    int(adapter._traci.edge.getLastStepHaltingNumber(edge_id))
                    for edge_id in adapter._traci.edge.getIDList()
                    if not edge_id.startswith(":")
                ),
            )
            if round(snapshot.time * 10) % 10 == 0:
                for vehicle_id in adapter._traci.vehicle.getIDList():
                    vehicle_time_loss[vehicle_id] = max(
                        vehicle_time_loss.get(vehicle_id, 0.0),
                        float(adapter._traci.vehicle.getTimeLoss(vehicle_id)),
                    )
                    leader = adapter._traci.vehicle.getLeader(vehicle_id, 100.0)
                    frames.append(
                        TrajectoryFrame(
                            time=snapshot.time,
                            follower_id=vehicle_id,
                            leader_id=leader[0] if leader else None,
                            gap=float(leader[1]) if leader else None,
                            follower_speed=max(0.0, float(adapter._traci.vehicle.getSpeed(vehicle_id))),
                            leader_speed=(
                                max(0.0, float(adapter._traci.vehicle.getSpeed(leader[0])))
                                if leader
                                else None
                            ),
                            follower_acceleration=float(
                                adapter._traci.vehicle.getAcceleration(vehicle_id)
                            ),
                        )
                    )
            if snapshot.time >= 1200:
                break
    finally:
        adapter.close()
    observations = []
    edge_positions = {f"e{index}": index * 76.54 for index in range(8)}
    for snapshot in snapshots:
        if round(snapshot.time * 10) % 10 != 0:
            continue
        for edge_id, edge in snapshot.edges.items():
            if edge_id in edge_positions:
                observations.append(
                    DetectorObservation(
                        time=snapshot.time,
                        position=edge_positions[edge_id],
                        density=edge.density_vehicles_per_km_per_lane,
                        speed=edge.mean_speed_meters_per_second,
                    )
                )
    detector = PhantomJamEventDetector(
        critical_density=25.0,
        low_speed_threshold=9.0,
        minimum_duration=3.0,
        minimum_amplitude=0.5,
    )
    events = detector.detect(observations)
    ssm_path = SCENARIO / "ssm.xml"
    fcd_path = SCENARIO / "fcd.xml"
    if not ssm_path.is_file() or not fcd_path.is_file():
        raise SystemExit("SUMO run incomplete: trajectory or SSM output is missing")
    conflicts = parse_sumo_ssm(str(ssm_path))
    safety = summarize_safety(
        frames,
        pet_values=[item.min_pet for item in conflicts if item.min_pet is not None],
    )
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    safety_payload = asdict(safety)
    distributions = {
        "ttc_seconds": safety_payload.pop("ttc_values"),
        "drac_meters_per_second2": safety_payload.pop("drac_values"),
        "pet_seconds": safety_payload.pop("pet_values"),
    }
    distribution_path = ARTIFACT / "safety_distributions.json"
    distribution_path.write_text(
        json.dumps(distributions, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    output = ARTIFACT / "summary.json"
    payload = {
        "complete": True,
        "seed": 42,
        "simulator_version": SumoAdapter.simulator_version(binary),
        "steps": len(snapshots),
        "last_time_seconds": snapshots[-1].time if snapshots else 0.0,
        "trajectory_frames": len(frames),
        "traffic": {
            "departed_vehicles": departed_vehicles,
            "arrived_vehicles": arrived_vehicles,
            "throughput_vehicles": arrived_vehicles,
            "maximum_halting_vehicles": maximum_halting_vehicles,
            "total_lost_time_seconds": float(sum(vehicle_time_loss.values())),
            "mean_edge_speed_meters_per_second": float(
                np.mean(
                    [
                        edge.mean_speed_meters_per_second
                        for snapshot in snapshots
                        for edge in snapshot.edges.values()
                    ]
                )
            ),
        },
        "phantom_event_count": len(events),
        "phantom_events": [asdict(event) for event in events],
        "ssm_conflict_count": len(conflicts),
        "ssm_conflict_types": summarize_ssm_conflict_types(conflicts),
        "safety": safety_payload,
        "safety_rates": {
            "ttc_conflicts_per_observation": safety.ttc_conflicts
            / max(1, safety.observation_count),
            "hard_braking_events_per_observation": safety.hard_braking_events
            / max(1, safety.observation_count),
            "high_closing_speed_conflicts_per_observation": (
                safety.high_closing_speed_conflicts / max(1, safety.observation_count)
            ),
        },
        "safety_distribution_summary": {
            name: distribution_summary(
                values,
                risk_tail=("lower" if name in {"ttc_seconds", "pet_seconds"} else "upper"),
            )
            for name, values in distributions.items()
        },
        "ssm_extrema": {
            "min_ttc_seconds": min(
                (item.min_ttc for item in conflicts if item.min_ttc is not None),
                default=None,
            ),
            "min_pet_seconds": min(
                (item.min_pet for item in conflicts if item.min_pet is not None),
                default=None,
            ),
            "max_drac_meters_per_second2": max(
                (item.max_drac for item in conflicts if item.max_drac is not None),
                default=None,
            ),
        },
        "output_hashes": {
            "fcd_sha256": checksum(fcd_path),
            "ssm_sha256": checksum(ssm_path),
            "safety_distributions_sha256": checksum(distribution_path),
        },
        "claim_boundary": (
            "synthetic microscopic ring fixture; surrogate conflicts are not crash probability"
        ),
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ended = datetime.now(timezone.utc)
    run_dir = ExperimentRegistry(str(ROOT / "artifacts" / "runs")).create(
        {"scenario": "sumo_ring_smoke", "seed": 42, "seeds": [42]},
        payload,
        simulator_version=payload["simulator_version"],
        input_paths=(
            str(SCENARIO / "ring.nod.xml"),
            str(SCENARIO / "ring.edg.xml"),
            str(SCENARIO / "ring.rou.xml"),
            str(SCENARIO / "ring.sumocfg"),
        ),
        external_output_paths=(str(output), str(distribution_path)),
        started_at=started,
        ended_at=ended,
        source_commit=source_commit,
        source_dirty=source_dirty,
    )
    shutil.copyfile(run_dir / "manifest.json", ARTIFACT / "manifest.json")
    print(output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.build_only:
        print(build_network())
    else:
        run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
