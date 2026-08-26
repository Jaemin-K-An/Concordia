from __future__ import annotations

import hashlib
import math
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from concordia.behavior import AcceptanceModel
from concordia.models import Route
from concordia.populations import generate_population
from concordia.preferences import UtilityModel, preference_slack
from concordia.simulation import SumoAdapter
from concordia.v9.action_space import AdaptiveAction
from v9_micro_sim import (
    ALL_INTERNAL_EDGES,
    _action_dict,
    _plan_action,
    _route_features,
    build_v9_network,
)
from v9_micro_sim import TrajectoryFrame, summarize_safety


BOTTLENECK_EDGES = ("m3", "a2", "b2", "out")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_run_files(
    directory: Path,
    network: Path,
    seed: int,
    demand: int,
    generation_seconds: int,
    reaction_multiplier: float,
):
    route_file = directory / "routes.rou.xml"
    spacing = 3600.0 / demand
    count = int(math.floor(generation_seconds / spacing))
    vehicles = "\n".join(
        f' <vehicle id="v{index:04d}" type="car" route="main" depart="{index * spacing:.3f}"/>'
        for index in range(count)
    )
    route_file.write_text(
        "<routes>\n"
        f' <vType id="car" accel="2.6" decel="4.5" emergencyDecel="8.0" sigma="0.45" tau="{reaction_multiplier:.5f}" speedDev="0.10"/>\n'
        ' <route id="main" edges="in m0 m1 m2 m3 out"/>\n'
        ' <route id="alternate1" edges="in a0 a1 a2 out"/>\n'
        ' <route id="alternate2" edges="in b0 b1 b2 out"/>\n'
        + vehicles
        + "\n</routes>\n",
        encoding="utf-8",
    )
    run_config = directory / "run.sumocfg"
    run_config.write_text(
        f"""<configuration>
 <input><net-file value="{network}"/><route-files value="{route_file}"/></input>
 <time><step-length value="1"/></time>
 <processing><collision.action value="warn"/><time-to-teleport value="-1"/></processing>
 <report><no-step-log value="true"/><duration-log.disable value="true"/></report>
 <random_number><seed value="{seed}"/></random_number>
</configuration>\n""",
        encoding="utf-8",
    )
    return run_config, route_file, count


def _acceptance_draw(future_seed: int, vehicle_id: str) -> float:
    return int(
        hashlib.sha256(f"{future_seed}::{vehicle_id}".encode()).hexdigest()[:13], 16
    ) / float(16**13)


def run_v10_candidate(
    network: Path,
    metadata: Mapping[str, object],
    config: Mapping[str, object],
    condition: Mapping[str, object],
    predecision_seed: int,
    future_seed: int,
    action: AdaptiveAction | Mapping[str, object],
    horizon_seconds: int,
    *,
    plan_actions: Sequence[AdaptiveAction | Mapping[str, object]] | None = None,
    demand_multiplier: float = 1.0,
    reaction_multiplier: float = 1.0,
) -> dict:
    if horizon_seconds <= 0:
        raise ValueError("v10 rollout horizon must be positive")
    action_value = _action_dict(action)
    realized_condition = dict(condition)
    realized_condition["demand"] = max(1, int(round(float(condition["demand"]) * demand_multiplier)))
    decision_time = float(config["decision_time_seconds"])
    stop_time = decision_time + float(horizon_seconds)
    with tempfile.TemporaryDirectory(prefix="concordia-v10-run-") as temporary:
        directory = Path(temporary)
        run_config, route_file, expected = _write_run_files(
            directory, network, predecision_seed, int(realized_condition["demand"]),
            int(config["vehicle_generation_seconds"]), reaction_multiplier,
        )
        binary = SumoAdapter.resolve_binary("sumo")
        if binary is None:
            raise RuntimeError("SUMO is unavailable")
        adapter = SumoAdapter(str(run_config), binary=binary)
        users = generate_population(
            expected, "s", "d", str(realized_condition["heterogeneity"]),
            float(config["preference_epsilon"]), 5.0, predecision_seed,
        )
        user_by_vehicle = {f"v{index:04d}": user for index, user in enumerate(users)}
        active_ids = {
            f"v{index:04d}" for index in range(expected)
            if ((index * 2654435761 + predecision_seed) % 10_000)
            < int(float(realized_condition["penetration"]) * 10_000)
        }
        utility = UtilityModel()
        acceptance_model = AcceptanceModel()
        departures = {}
        completed_travel_time = 0.0
        arrived = 0
        vehicle_time = 0.0
        queue_integral = 0.0
        bottleneck_integral = 0.0
        frames = []
        maximum_regret = 0.0
        offers = accepted = rejected = 0
        planned = None
        candidate_plans = {}
        adapter.start(predecision_seed)
        try:
            while (
                adapter._traci.simulation.getMinExpectedNumber() > 0
                and adapter._traci.simulation.getTime() < stop_time
            ):
                snapshot = adapter.step()
                now = float(snapshot.time)
                if planned is None and now >= decision_time:
                    routes_at_decision = _route_features(adapter)
                    planned = _plan_action(
                        action_value, users, active_ids, routes_at_decision,
                        metadata, realized_condition, int(config["vehicle_generation_seconds"]),
                    )
                    for item in plan_actions or ():
                        item_value = _action_dict(item)
                        plan = _plan_action(
                            item_value, users, active_ids, routes_at_decision,
                            metadata, realized_condition,
                            int(config["vehicle_generation_seconds"]),
                        )
                        plan.update({
                            "route_mapping_valid": True,
                            "all_routes_legal": bool(metadata["all_routes_legal"]),
                            "preference_feasible": True,
                        })
                        candidate_plans[str(item_value["action_id"])] = plan
                for vehicle_id in adapter._traci.simulation.getDepartedIDList():
                    departures[vehicle_id] = now
                    if vehicle_id not in active_ids:
                        continue
                    routes: tuple[Route, ...] = _route_features(adapter)
                    fastest = min(routes, key=lambda route: route.features.time)
                    if now <= decision_time or bool(action_value.get("is_null", False)):
                        if fastest.route_id != "main":
                            adapter._traci.vehicle.setRoute(vehicle_id, list(fastest.nodes))
                        continue
                    if planned is None or vehicle_id not in planned["assignments"]:
                        if fastest.route_id != "main":
                            adapter._traci.vehicle.setRoute(vehicle_id, list(fastest.nodes))
                        continue
                    route_index = int(planned["assignments"][vehicle_id]) + 1
                    target = routes[route_index]
                    user = user_by_vehicle[vehicle_id]
                    utilities = utility.utilities(user.preferences, routes)
                    slack = float(preference_slack(utilities)[target.route_id])
                    if slack > user.epsilon + 1e-12:
                        continue
                    offers += 1
                    probability = acceptance_model.probability(
                        slack,
                        utilities[target.route_id] - utilities["main"],
                        routes[0].features.time - target.features.time,
                        routes[0].features.variability - target.features.variability,
                        max(0.0, routes[0].features.time - target.features.time),
                    )
                    probability = float(np.clip(
                        probability * float(realized_condition["acceptance_multiplier"]),
                        0.0, 1.0,
                    ))
                    if _acceptance_draw(future_seed, vehicle_id) <= probability:
                        adapter._traci.vehicle.setRoute(vehicle_id, list(target.nodes))
                        maximum_regret = max(maximum_regret, slack)
                        accepted += 1
                    else:
                        rejected += 1
                for vehicle_id in adapter._traci.simulation.getArrivedIDList():
                    if vehicle_id in departures:
                        completed_travel_time += now - departures[vehicle_id]
                        arrived += 1
                if now <= decision_time:
                    continue
                vehicle_ids = list(adapter._traci.vehicle.getIDList())
                vehicle_time += len(vehicle_ids)
                queue_integral += sum(
                    int(adapter._traci.edge.getLastStepHaltingNumber(edge))
                    for edge in ALL_INTERNAL_EDGES
                )
                bottleneck_integral += sum(
                    int(adapter._traci.edge.getLastStepVehicleNumber(edge))
                    for edge in BOTTLENECK_EDGES
                ) / len(BOTTLENECK_EDGES)
                for vehicle_id in vehicle_ids:
                    speed = max(0.0, float(adapter._traci.vehicle.getSpeed(vehicle_id)))
                    acceleration = float(adapter._traci.vehicle.getAcceleration(vehicle_id))
                    leader = adapter._traci.vehicle.getLeader(vehicle_id, 150.0)
                    gap = max(1e-6, float(leader[1])) if leader else None
                    leader_speed = (
                        max(0.0, float(adapter._traci.vehicle.getSpeed(leader[0])))
                        if leader else None
                    )
                    frames.append(TrajectoryFrame(
                        now, vehicle_id, leader[0] if leader else None,
                        gap, speed, leader_speed, acceleration,
                    ))
        finally:
            adapter.close()
        if planned is None:
            raise RuntimeError("v10 decision state was not observed")
        safety = asdict(summarize_safety(frames))
        for name in ("ttc_values", "drac_values", "pet_values"):
            safety.pop(name)
        return {
            "state_seed": predecision_seed,
            "future_seed": future_seed,
            "action_id": str(action_value["action_id"]),
            "horizon_seconds": horizon_seconds,
            "condition": realized_condition,
            "action_plan": planned,
            "candidate_plans": candidate_plans,
            "generated_vehicle_count": expected,
            "arrived_vehicle_count": arrived,
            "completed_travel_time_seconds": completed_travel_time,
            "vehicle_time_seconds": vehicle_time,
            "queue_integral": queue_integral,
            "bottleneck_load_integral": bottleneck_integral,
            "safety": safety,
            "maximum_affected_regret": maximum_regret,
            "all_executed_routes_legal": bool(metadata["all_routes_legal"]),
            "offer_count": offers,
            "accepted_count": accepted,
            "rejected_count": rejected,
            "network_hash": _sha(network),
            "route_file_hash": _sha(route_file),
            "reaction_multiplier": reaction_multiplier,
            "demand_multiplier": demand_multiplier,
        }


def paired_rollout_result(baseline: Mapping[str, object], action: Mapping[str, object]) -> dict:
    if (
        baseline["state_seed"] != action["state_seed"]
        or baseline["future_seed"] != action["future_seed"]
        or baseline["network_hash"] != action["network_hash"]
        or baseline["route_file_hash"] != action["route_file_hash"]
    ):
        raise RuntimeError("v10 paired rollout contract failed")
    baseline_time = float(baseline["vehicle_time_seconds"])
    baseline_queue = float(baseline["queue_integral"])
    baseline_bottleneck = float(baseline["bottleneck_load_integral"])
    return {
        "traffic_gain": (
            baseline_time - float(action["vehicle_time_seconds"])
        ) / max(baseline_time, 1e-9),
        "queue_delta": (
            float(action["queue_integral"]) - baseline_queue
        ) / max(baseline_queue, 1.0),
        "risk_delta": (
            float(action["safety"]["cvar_drac_95"])
            - float(baseline["safety"]["cvar_drac_95"])
        ),
        "bottleneck_load_delta": (
            float(action["bottleneck_load_integral"]) - baseline_bottleneck
        ) / max(baseline_bottleneck, 1.0),
        "maximum_regret": float(action["maximum_affected_regret"]),
        "legal": bool(action["all_executed_routes_legal"]),
    }


__all__ = [
    "build_v9_network",
    "paired_rollout_result",
    "run_v10_candidate",
]
