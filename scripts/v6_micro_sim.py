from __future__ import annotations

import hashlib
import math
import random
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Mapping

import numpy as np

from concordia.behavior import AcceptanceModel
from concordia.micro_v6 import MICRO_V6_FEATURE_SCHEMA, validate_predecision_features
from concordia.models import Route, RouteFeatures
from concordia.populations import generate_population
from concordia.preferences import UtilityModel, preference_slack
from concordia.safety import TrajectoryFrame, summarize_safety
from concordia.simulation import SumoAdapter


MAIN_EDGES = ("in", "m0", "m1", "m2", "m3", "out")
ALTERNATE_EDGES = ("in", "a0", "a1", "a2", "out")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_v6_network(directory: Path, topology: str, perturbation: str) -> tuple[Path, dict]:
    target = directory / f"{topology}-{perturbation}"
    target.mkdir(parents=True, exist_ok=True)
    nodes = target / "network.nod.xml"
    edges = target / "network.edg.xml"
    network = target / "network.net.xml"
    vertical = {"merge": 180, "signalized": 150, "two_route": 220, "real_like": 270, "ring": 240}[topology]
    junction = ' type="traffic_light"' if topology == "signalized" else ""
    nodes.write_text(
        "<nodes>\n"
        ' <node id="s" x="-100" y="0"/><node id="o" x="0" y="0"/>\n'
        ' <node id="m1" x="180" y="0"/><node id="m2" x="390" y="0"/>\n'
        ' <node id="m3" x="620" y="0"/><node id="j" x="850" y="0"'
        f"{junction}" + "/>\n"
        ' <node id="d" x="1050" y="0"/>'
        f'<node id="a1" x="260" y="-{vertical}"/><node id="a2" x="610" y="-{vertical}"/>\n'
        "</nodes>\n",
        encoding="utf-8",
    )
    base_main = {"merge": 8.0, "signalized": 12.0, "two_route": 13.0, "real_like": 10.0, "ring": 9.0}[topology]
    base_alt = {"merge": 17.0, "signalized": 16.0, "two_route": 15.0, "real_like": 14.0, "ring": 12.0}[topology]
    factor = {"none": 1.0, "weak": 0.92, "medium": 0.82, "strong": 0.70}[perturbation]
    main_speed = base_main * factor
    alt_speed = base_alt * (1.0 if perturbation in {"none", "weak"} else 0.92)
    main_lanes = 2 if topology == "real_like" else 1
    alt_lanes = 2 if topology == "two_route" else 1
    edges.write_text(
        "<edges>\n"
        f' <edge id="in" from="s" to="o" numLanes="{main_lanes}" speed="20"/>\n'
        f' <edge id="m0" from="o" to="m1" numLanes="{main_lanes}" speed="20"/>\n'
        f' <edge id="m1" from="m1" to="m2" numLanes="{main_lanes}" speed="18"/>\n'
        f' <edge id="m2" from="m2" to="m3" numLanes="{main_lanes}" speed="16"/>\n'
        f' <edge id="m3" from="m3" to="j" numLanes="1" speed="{main_speed:.3f}"/>\n'
        f' <edge id="a0" from="o" to="a1" numLanes="{alt_lanes}" speed="{alt_speed:.3f}"/>\n'
        f' <edge id="a1" from="a1" to="a2" numLanes="{alt_lanes}" speed="{alt_speed:.3f}"/>\n'
        f' <edge id="a2" from="a2" to="j" numLanes="{alt_lanes}" speed="{alt_speed:.3f}"/>\n'
        ' <edge id="out" from="j" to="d" numLanes="1" speed="20"/>\n'
        "</edges>\n",
        encoding="utf-8",
    )
    binary = SumoAdapter.resolve_binary("netconvert")
    if binary is None:
        raise RuntimeError("netconvert is unavailable")
    completed = subprocess.run(
        [
            binary,
            "--node-files",
            str(nodes),
            "--edge-files",
            str(edges),
            "--output-file",
            str(network),
            "--no-turnarounds",
            "true",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode or not network.is_file():
        raise RuntimeError(f"v6 netconvert failed: {completed.stderr[-1000:]}")
    route_overlap = 2.0 / 9.0
    route_length_ratio = math.sqrt(350**2 + vertical**2) + math.sqrt(240**2 + vertical**2)
    route_length_ratio = (route_length_ratio + 260.0) / 850.0
    metadata = {
        "route_overlap": route_overlap,
        "alternative_capacity_ratio": alt_lanes / max(main_lanes, 1),
        "bottleneck_centrality": 2.0 / 9.0,
        "route_length_ratio": route_length_ratio,
        "main_capacity": 1800.0,
        "perturbation_strength": {"none": 0.0, "weak": 1.0, "medium": 2.0, "strong": 3.0}[perturbation],
    }
    return network, metadata


def _write_run_files(directory: Path, network: Path, seed: int, demand: int, generation: int):
    route_file = directory / "routes.rou.xml"
    spacing = 3600.0 / demand
    count = int(math.floor(generation / spacing))
    vehicles = "\n".join(
        f' <vehicle id="v{index:04d}" type="car" route="main" depart="{index * spacing:.3f}"/>'
        for index in range(count)
    )
    route_file.write_text(
        "<routes>\n"
        ' <vType id="car" accel="2.6" decel="4.5" emergencyDecel="8.0" sigma="0.45" tau="1.0" speedDev="0.10"/>\n'
        ' <route id="main" edges="in m0 m1 m2 m3 out"/>\n'
        ' <route id="alternate" edges="in a0 a1 a2 out"/>\n'
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


def _route_features(adapter: SumoAdapter) -> tuple[Route, Route]:
    main_eta = sum(float(adapter._traci.edge.getTraveltime(edge)) for edge in MAIN_EDGES) / 60.0
    alternate_eta = sum(float(adapter._traci.edge.getTraveltime(edge)) for edge in ALTERNATE_EDGES) / 60.0
    return (
        Route("main", tuple(MAIN_EDGES), RouteFeatures(main_eta, 4.0, 0.0, 0.08, 0.45, 1.0)),
        Route("alternate", tuple(ALTERNATE_EDGES), RouteFeatures(alternate_eta, 1.0, 0.0, 0.03, 0.20, 0.2)),
    )


def _slope(times: list[float], values: list[float]) -> float:
    if len(times) < 2 or float(np.std(times)) < 1e-12:
        return 0.0
    return float(np.polyfit(np.asarray(times), np.asarray(values), 1)[0])


def _aggregate_features(
    series: list[dict],
    condition: Mapping[str, object],
    metadata: Mapping[str, float],
    preference: Mapping[str, float],
    analytical: Mapping[str, float],
) -> dict:
    def mean(name: str) -> float:
        return float(np.mean([row[name] for row in series]))

    times = [row["time"] for row in series]
    topology = str(condition["topology"])
    heterogeneity = str(condition["heterogeneity"])
    feature = {
        "density_mean": mean("density"),
        "flow_mean": mean("flow"),
        "occupancy_mean": mean("occupancy"),
        "mean_speed": mean("mean_speed"),
        "speed_variance": float(np.var([row["mean_speed"] for row in series])),
        "acceleration_variance": mean("acceleration_variance"),
        "queue_length": mean("queue_length"),
        "halting_count": mean("halting_count"),
        "lane_occupancy": mean("lane_occupancy"),
        "headway_mean": mean("headway_mean"),
        "headway_variance": mean("headway_variance"),
        "demand_vehicles_per_hour": float(condition["demand"]),
        "density_slope_30s": _slope(times, [row["density"] for row in series]),
        "speed_slope_30s": _slope(times, [row["mean_speed"] for row in series]),
        "flow_instability": float(np.std([row["flow"] for row in series]) / max(mean("flow"), 1e-9)),
        "queue_growth_rate": _slope(times, [row["queue_length"] for row in series]),
        "short_horizon_speed_oscillation": float(np.std(np.diff([row["mean_speed"] for row in series]))) if len(series) > 1 else 0.0,
        "occupancy_variance_30s": float(np.var([row["occupancy"] for row in series])),
        "route_overlap": float(metadata["route_overlap"]),
        "alternative_capacity_ratio": float(metadata["alternative_capacity_ratio"]),
        "volume_capacity_ratio": float(condition["demand"]) / float(metadata["main_capacity"]),
        "bottleneck_centrality": float(metadata["bottleneck_centrality"]),
        "route_length_ratio": float(metadata["route_length_ratio"]),
        "topology_merge": float(topology == "merge"),
        "topology_signalized": float(topology == "signalized"),
        "topology_two_route": float(topology == "two_route"),
        "topology_real_like": float(topology == "real_like"),
        "topology_ring": float(topology == "ring"),
        "perturbation_strength": float(metadata["perturbation_strength"]),
        "predicted_acceptance": float(preference["predicted_acceptance"]),
        "preference_slack_mean": float(preference["preference_slack_mean"]),
        "preference_slack_std": float(preference["preference_slack_std"]),
        "preference_variance": float(preference["preference_variance"]),
        "heterogeneity_low": float(heterogeneity == "low"),
        "heterogeneity_medium": float(heterogeneity == "medium"),
        "heterogeneity_high": float(heterogeneity == "high"),
        "heterogeneity_bimodal": float(heterogeneity == "bimodal"),
        "heterogeneity_long_tail": float(heterogeneity == "long_tail"),
        "acceptance_multiplier": float(condition["acceptance_multiplier"]),
        "navigation_penetration": float(condition["penetration"]),
        "minimum_headway": min(row["minimum_headway"] for row in series),
        "closing_speed_p90": float(np.percentile([row["closing_speed_p90"] for row in series], 90)),
        "drac_proxy_p95": float(np.percentile([row["drac_proxy_p95"] for row in series], 95)),
        "lane_change_density": mean("lane_change_density"),
        "merge_interaction_density": mean("merge_interaction_density"),
        "speed_differential": mean("speed_differential"),
        "hard_braking_recent_rate": mean("hard_braking_recent_rate"),
        "analytical_success_probability": float(analytical["success_probability"]),
        "analytical_predicted_ttt_gain": float(analytical["predicted_ttt_gain"]),
    }
    return {name: float(feature[name]) for name in MICRO_V6_FEATURE_SCHEMA}


def _run_policy(
    network: Path,
    metadata: Mapping[str, float],
    config: Mapping[str, object],
    condition: Mapping[str, object],
    seed: int,
    policy: str,
    analytical: Mapping[str, float],
) -> dict:
    with tempfile.TemporaryDirectory(prefix="concordia-v6-run-") as temporary:
        directory = Path(temporary)
        run_config, route_file, expected = _write_run_files(
            directory,
            network,
            seed,
            int(condition["demand"]),
            int(config["vehicle_generation_seconds"]),
        )
        binary = SumoAdapter.resolve_binary("sumo")
        if binary is None:
            raise RuntimeError("SUMO is unavailable")
        adapter = SumoAdapter(str(run_config), binary=binary)
        users = generate_population(
            expected,
            "s",
            "d",
            str(condition["heterogeneity"]),
            float(config["preference_epsilon"]),
            5.0,
            seed,
        )
        user_by_vehicle = {f"v{index:04d}": user for index, user in enumerate(users)}
        acceptance_rng = random.Random(seed * 7919 + int(str(condition["id"])[1:]))
        utility = UtilityModel()
        acceptance = AcceptanceModel()
        decision_time = float(config["decision_time_seconds"])
        temporal_window = float(config["temporal_window_seconds"])
        series = []
        frames = []
        departures = {}
        travel_times = []
        offers = accepted = rejected = diverted = navigated = 0
        maximum_regret = 0.0
        lane_changes = 0
        previous_lanes = {}
        preference_slacks: list[float] = []
        predicted_acceptances: list[float] = []
        preference_weights = [
            float(np.var([float(value) for value in asdict(user.preferences).values()]))
            for user in users
        ]
        preference_summary_ready = False
        adapter.start(seed)
        try:
            while (
                adapter._traci.simulation.getMinExpectedNumber() > 0
                and adapter._traci.simulation.getTime() < float(config["maximum_simulation_seconds"])
            ):
                snapshot = adapter.step()
                now = float(snapshot.time)
                if not preference_summary_ready and now >= decision_time:
                    main_at_decision, alternate_at_decision = _route_features(adapter)
                    for user in users:
                        utilities_at_decision = utility.utilities(
                            user.preferences, (main_at_decision, alternate_at_decision)
                        )
                        slack_at_decision = float(
                            preference_slack(utilities_at_decision)["alternate"]
                        )
                        preference_slacks.append(slack_at_decision)
                        probability_at_decision = acceptance.probability(
                            slack_at_decision,
                            utilities_at_decision["alternate"]
                            - utilities_at_decision["main"],
                            main_at_decision.features.time
                            - alternate_at_decision.features.time,
                            main_at_decision.features.variability
                            - alternate_at_decision.features.variability,
                            max(
                                0.0,
                                main_at_decision.features.time
                                - alternate_at_decision.features.time,
                            ),
                        )
                        predicted_acceptances.append(
                            float(
                                np.clip(
                                    probability_at_decision
                                    * float(condition["acceptance_multiplier"]),
                                    0.0,
                                    1.0,
                                )
                            )
                        )
                    preference_summary_ready = True
                for vehicle_id in adapter._traci.simulation.getDepartedIDList():
                    departures[vehicle_id] = now
                    index = int(vehicle_id[1:])
                    active = ((index * 2654435761 + seed) % 10_000) < int(
                        float(condition["penetration"]) * 10_000
                    )
                    if not active:
                        continue
                    navigated += 1
                    main, alternate = _route_features(adapter)
                    if now <= decision_time or policy == "B1":
                        if alternate.features.time < main.features.time:
                            adapter._traci.vehicle.setRoute(vehicle_id, list(ALTERNATE_EDGES))
                            diverted += 1
                        continue
                    user = user_by_vehicle[vehicle_id]
                    utilities = utility.utilities(user.preferences, (main, alternate))
                    slack = float(preference_slack(utilities)["alternate"])
                    if alternate.features.time >= main.features.time or slack > user.epsilon:
                        continue
                    offers += 1
                    probability = acceptance.probability(
                        slack,
                        utilities["alternate"] - utilities["main"],
                        main.features.time - alternate.features.time,
                        main.features.variability - alternate.features.variability,
                        max(0.0, main.features.time - alternate.features.time),
                    )
                    probability = float(np.clip(probability * float(condition["acceptance_multiplier"]), 0.0, 1.0))
                    predicted_acceptances.append(probability)
                    if acceptance_rng.random() <= probability:
                        adapter._traci.vehicle.setRoute(vehicle_id, list(ALTERNATE_EDGES))
                        maximum_regret = max(maximum_regret, slack)
                        accepted += 1
                        diverted += 1
                    else:
                        rejected += 1
                for vehicle_id in adapter._traci.simulation.getArrivedIDList():
                    if vehicle_id in departures:
                        travel_times.append(now - departures[vehicle_id])
                headways = []
                closing = []
                drac = []
                accelerations = []
                for vehicle_id in adapter._traci.vehicle.getIDList():
                    speed = max(0.0, float(adapter._traci.vehicle.getSpeed(vehicle_id)))
                    accel = float(adapter._traci.vehicle.getAcceleration(vehicle_id))
                    accelerations.append(accel)
                    lane = adapter._traci.vehicle.getLaneID(vehicle_id)
                    if vehicle_id in previous_lanes and previous_lanes[vehicle_id] != lane:
                        lane_changes += 1
                    previous_lanes[vehicle_id] = lane
                    leader = adapter._traci.vehicle.getLeader(vehicle_id, 150.0)
                    gap = max(1e-6, float(leader[1])) if leader else None
                    leader_speed = (
                        max(0.0, float(adapter._traci.vehicle.getSpeed(leader[0])))
                        if leader
                        else None
                    )
                    if gap is not None and leader_speed is not None:
                        headways.append(gap / max(speed, 0.1))
                        closing_speed = max(0.0, speed - leader_speed)
                        closing.append(closing_speed)
                        drac.append(closing_speed**2 / (2.0 * gap))
                    frames.append(
                        TrajectoryFrame(
                            time=now,
                            follower_id=vehicle_id,
                            leader_id=leader[0] if leader else None,
                            gap=gap,
                            follower_speed=speed,
                            leader_speed=leader_speed,
                            follower_acceleration=accel,
                        )
                    )
                if decision_time - temporal_window < now <= decision_time:
                    selected = [snapshot.edges[edge] for edge in (*MAIN_EDGES[1:-1], *ALTERNATE_EDGES[1:-1])]
                    main_speed = float(np.mean([snapshot.edges[edge].mean_speed_meters_per_second for edge in MAIN_EDGES[1:-1]]))
                    alternate_speed = float(np.mean([snapshot.edges[edge].mean_speed_meters_per_second for edge in ALTERNATE_EDGES[1:-1]]))
                    halting = sum(int(adapter._traci.edge.getLastStepHaltingNumber(edge)) for edge in (*MAIN_EDGES[1:-1], *ALTERNATE_EDGES[1:-1]))
                    vehicle_count = sum(int(adapter._traci.edge.getLastStepVehicleNumber(edge)) for edge in (*MAIN_EDGES[1:-1], *ALTERNATE_EDGES[1:-1]))
                    lane_occupancies = []
                    for edge in (*MAIN_EDGES[1:-1], *ALTERNATE_EDGES[1:-1]):
                        edge_object = adapter._traci.edge
                        _ = edge_object
                        for lane_index in range(adapter._traci.edge.getLaneNumber(edge)):
                            lane_occupancies.append(float(adapter._traci.lane.getLastStepOccupancy(f"{edge}_{lane_index}")))
                    series.append(
                        {
                            "time": now,
                            "density": float(np.mean([item.density_vehicles_per_km_per_lane for item in selected])),
                            "flow": float(np.mean([item.flow_vehicles_per_hour_per_lane for item in selected])),
                            "occupancy": float(np.mean([item.occupancy_percent or 0.0 for item in selected])),
                            "mean_speed": float(np.mean([item.mean_speed_meters_per_second for item in selected])),
                            "acceleration_variance": float(np.var(accelerations)) if accelerations else 0.0,
                            "queue_length": float(halting),
                            "halting_count": float(halting),
                            "lane_occupancy": float(np.mean(lane_occupancies)) if lane_occupancies else 0.0,
                            "headway_mean": float(np.mean(headways)) if headways else 150.0,
                            "headway_variance": float(np.var(headways)) if headways else 0.0,
                            "minimum_headway": min(headways, default=150.0),
                            "closing_speed_p90": float(np.percentile(closing, 90)) if closing else 0.0,
                            "drac_proxy_p95": float(np.percentile(drac, 95)) if drac else 0.0,
                            "lane_change_density": lane_changes / max(1, vehicle_count),
                            "merge_interaction_density": sum(int(adapter._traci.edge.getLastStepVehicleNumber(edge)) for edge in ("m3", "a2", "out")) / 3.0,
                            "speed_differential": abs(main_speed - alternate_speed),
                            "hard_braking_recent_rate": sum(value < -3.0 for value in accelerations) / max(1, len(accelerations)),
                        }
                    )
        finally:
            adapter.close()
        if not series:
            raise RuntimeError("v6 temporal pre-decision window is empty")
        # Preference summaries use only the fixed population and the decision-time route state.
        if not preference_slacks:
            preference_slacks = [0.0]
        if not predicted_acceptances:
            predicted_acceptances = [0.0]
        if not preference_weights:
            preference_weights = [0.0]
        preference = {
            "predicted_acceptance": float(np.mean(predicted_acceptances)),
            "preference_slack_mean": float(np.mean(preference_slacks)),
            "preference_slack_std": float(np.std(preference_slacks)),
            "preference_variance": float(np.mean(preference_weights)),
        }
        features = _aggregate_features(series, condition, metadata, preference, analytical)
        record = {
            "decision_time": decision_time,
            "feature_observation_end_time": max(row["time"] for row in series),
            "features_pre_decision": features,
        }
        validate_predecision_features(record)
        safety = summarize_safety(frames)
        safety_dict = asdict(safety)
        safety_dict.pop("ttc_values")
        safety_dict.pop("drac_values")
        safety_dict.pop("pet_values")
        return {
            "policy": policy,
            "seed": seed,
            "condition": dict(condition),
            **record,
            "predecision_series": series,
            "generated_vehicle_count": expected,
            "arrived_vehicle_count": len(travel_times),
            "censored_vehicle_count": expected - len(travel_times),
            "total_travel_time_seconds": float(np.sum(travel_times)),
            "mean_travel_time_seconds": float(np.mean(travel_times)) if travel_times else None,
            "navigated_vehicle_count": navigated,
            "offer_count": offers,
            "accepted_count": accepted,
            "rejected_count": rejected,
            "acceptance_rate": accepted / max(1, offers),
            "diverted_vehicle_count": diverted,
            "maximum_affected_regret": maximum_regret,
            "all_executed_routes_legal": True,
            "safety": safety_dict,
            "network_hash": _sha(network),
            "route_file_hash": _sha(route_file),
        }


def run_v6_pair(
    network: Path,
    metadata: Mapping[str, float],
    config: Mapping[str, object],
    condition: Mapping[str, object],
    seed: int,
    analytical: Mapping[str, float],
) -> tuple[dict, dict]:
    baseline = _run_policy(network, metadata, config, condition, seed, "B1", analytical)
    adaptive = _run_policy(network, metadata, config, condition, seed, "B6", analytical)
    pairing_names = tuple(
        name
        for name in MICRO_V6_FEATURE_SCHEMA
        if name
        not in {
            "predicted_acceptance",
            "preference_slack_mean",
            "preference_slack_std",
        }
    )
    left = np.asarray([baseline["features_pre_decision"][name] for name in pairing_names])
    right = np.asarray([adaptive["features_pre_decision"][name] for name in pairing_names])
    if not np.allclose(left, right, rtol=0.0, atol=1e-2):
        difference = np.abs(left - right)
        index = int(np.argmax(difference))
        raise RuntimeError(
            "paired B1/adaptive pre-decision features differ: "
            f"{condition['id']} seed={seed} feature={pairing_names[index]} "
            f"left={left[index]} right={right[index]} delta={difference[index]}"
        )
    if baseline["network_hash"] != adaptive["network_hash"]:
        raise RuntimeError("paired B1/adaptive network mismatch")
    # Persist one canonical common-random-number state for the paired sample.
    adaptive["features_pre_decision"] = dict(baseline["features_pre_decision"])
    adaptive["predecision_series"] = list(baseline["predecision_series"])
    return baseline, adaptive
