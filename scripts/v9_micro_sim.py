from __future__ import annotations

import hashlib
import math
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

import concordia.feasibility  # noqa: F401 - initializes the legacy feasibility/safety import cycle
from concordia.behavior import AcceptanceModel
from concordia.models import Route, RouteFeatures
from concordia.populations import generate_population
from concordia.preferences import UtilityModel, preference_slack
from concordia.safety.metrics import TrajectoryFrame, summarize_safety
from concordia.simulation import SumoAdapter
from concordia.v9.action_space import AdaptiveAction
from concordia.v9.route_allocation import (
    action_concentration_index,
    allocation_weights,
    deterministic_route_index,
    route_entropy,
)
from concordia.v9.user_selection import rank_feasible_users
from v6_micro_sim import _aggregate_features


MAIN_EDGES = ("in", "m0", "m1", "m2", "m3", "out")
ALT1_EDGES = ("in", "a0", "a1", "a2", "out")
ALT2_EDGES = ("in", "b0", "b1", "b2", "out")
ALL_INTERNAL_EDGES = (*MAIN_EDGES[1:-1], *ALT1_EDGES[1:-1], *ALT2_EDGES[1:-1])


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_v9_network(directory: Path, topology: str, perturbation: str) -> tuple[Path, dict]:
    target = directory / f"v9-{topology}-{perturbation}"
    target.mkdir(parents=True, exist_ok=True)
    nodes = target / "network.nod.xml"
    edges = target / "network.edg.xml"
    network = target / "network.net.xml"
    vertical = {"merge": 180, "signalized": 150, "two_route": 220, "asymmetric": 260, "real_like": 280}[topology]
    junction = ' type="traffic_light"' if topology == "signalized" else ""
    nodes.write_text(
        "<nodes>\n"
        ' <node id="s" x="-100" y="0"/><node id="o" x="0" y="0"/>\n'
        ' <node id="m1" x="180" y="0"/><node id="m2" x="390" y="0"/>\n'
        ' <node id="m3" x="620" y="0"/><node id="j" x="850" y="0"'
        f"{junction}" + "/>\n"
        ' <node id="d" x="1050" y="0"/>\n'
        f' <node id="a1" x="260" y="-{vertical}"/><node id="a2" x="610" y="-{vertical}"/>\n'
        f' <node id="b1" x="230" y="{vertical * 0.75:.1f}"/><node id="b2" x="590" y="{vertical * 1.10:.1f}"/>\n'
        "</nodes>\n",
        encoding="utf-8",
    )
    main_base = {"merge": 8.0, "signalized": 12.0, "two_route": 13.0, "asymmetric": 9.5, "real_like": 10.0}[topology]
    alt1_base = {"merge": 17.0, "signalized": 16.0, "two_route": 15.0, "asymmetric": 15.5, "real_like": 14.0}[topology]
    alt2_base = {"merge": 14.0, "signalized": 14.5, "two_route": 13.5, "asymmetric": 13.0, "real_like": 13.0}[topology]
    factor = {"none": 1.0, "weak": 0.92, "medium": 0.82, "strong": 0.70}[perturbation]
    main_speed = main_base * factor
    alt1_speed = alt1_base * (1.0 if perturbation in {"none", "weak"} else 0.92)
    alt2_speed = alt2_base * (1.0 if perturbation != "strong" else 0.88)
    main_lanes = 2 if topology in {"real_like", "asymmetric"} else 1
    alt1_lanes = 2 if topology == "two_route" else 1
    alt2_lanes = 2 if topology == "real_like" else 1
    edges.write_text(
        "<edges>\n"
        f' <edge id="in" from="s" to="o" numLanes="{main_lanes}" speed="20"/>\n'
        f' <edge id="m0" from="o" to="m1" numLanes="{main_lanes}" speed="20"/>\n'
        f' <edge id="m1" from="m1" to="m2" numLanes="{main_lanes}" speed="18"/>\n'
        f' <edge id="m2" from="m2" to="m3" numLanes="{main_lanes}" speed="16"/>\n'
        f' <edge id="m3" from="m3" to="j" numLanes="1" speed="{main_speed:.3f}"/>\n'
        f' <edge id="a0" from="o" to="a1" numLanes="{alt1_lanes}" speed="{alt1_speed:.3f}"/>\n'
        f' <edge id="a1" from="a1" to="a2" numLanes="{alt1_lanes}" speed="{alt1_speed:.3f}"/>\n'
        f' <edge id="a2" from="a2" to="j" numLanes="{alt1_lanes}" speed="{alt1_speed:.3f}"/>\n'
        f' <edge id="b0" from="o" to="b1" numLanes="{alt2_lanes}" speed="{alt2_speed:.3f}"/>\n'
        f' <edge id="b1" from="b1" to="b2" numLanes="{alt2_lanes}" speed="{alt2_speed:.3f}"/>\n'
        f' <edge id="b2" from="b2" to="j" numLanes="{alt2_lanes}" speed="{alt2_speed:.3f}"/>\n'
        ' <edge id="out" from="j" to="d" numLanes="1" speed="20"/>\n'
        "</edges>\n",
        encoding="utf-8",
    )
    binary = SumoAdapter.resolve_binary("netconvert")
    if binary is None:
        raise RuntimeError("netconvert is unavailable")
    completed = subprocess.run(
        [binary, "--node-files", str(nodes), "--edge-files", str(edges), "--output-file", str(network), "--no-turnarounds", "true"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode or not network.is_file():
        raise RuntimeError(f"v9 netconvert failed: {completed.stderr[-1000:]}")
    return network, {
        "route_overlap": 2.0 / 12.0,
        "alternative_capacity_ratio": (alt1_lanes + alt2_lanes) / max(main_lanes, 1),
        "bottleneck_centrality": 3.0 / 15.0,
        "route_length_ratio": 1.18 if topology != "asymmetric" else 1.32,
        "main_capacity": 1800.0 * main_lanes,
        "alternative_capacities": [1800.0 * alt1_lanes, 1800.0 * alt2_lanes],
        "perturbation_strength": {"none": 0.0, "weak": 1.0, "medium": 2.0, "strong": 3.0}[perturbation],
        "physical_route_alternatives": 3,
        "all_routes_legal": True,
    }


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


def _route_features(adapter: SumoAdapter) -> tuple[Route, Route, Route]:
    def route(name: str, edges: tuple[str, ...], variability: float, risk: float, complexity: float, familiarity: float):
        eta = sum(float(adapter._traci.edge.getTraveltime(edge)) for edge in edges) / 60.0
        return Route(name, edges, RouteFeatures(eta, variability, 0.0, risk, complexity, familiarity))
    return (
        route("main", MAIN_EDGES, 4.0, 0.08, 0.45, 1.0),
        route("alternate1", ALT1_EDGES, 1.0, 0.03, 0.20, 0.2),
        route("alternate2", ALT2_EDGES, 1.8, 0.04, 0.28, 0.35),
    )


def _action_dict(action: AdaptiveAction | Mapping[str, object]) -> dict:
    return action.to_dict() if isinstance(action, AdaptiveAction) else dict(action)


def _plan_action(
    action: Mapping[str, object],
    users,
    active_ids: set[str],
    routes: tuple[Route, Route, Route],
    metadata: Mapping[str, object],
    condition: Mapping[str, object],
    generation_seconds: int,
) -> dict:
    if bool(action.get("is_null", False)):
        return {
            "selected_user_ids": [],
            "assignments": {},
            "route_weights": [0.0, 0.0],
            **{name: 0.0 for name in (
                "proposed_rerouted_user_count", "expected_accepted_user_count", "expected_rerouted_flow",
                "maximum_target_edge_load_increase", "maximum_source_edge_load_reduction", "bottleneck_flow_delta",
                "destination_capacity_slack", "route_overlap_delta", "route_entropy", "affected_junction_count",
                "expected_lane_change_demand_delta", "conflict_zone_exposure_delta", "action_concentration_index",
                "average_preference_slack_selected", "p90_preference_slack_selected", "expected_acceptance_probability",
                "route_length_increase", "reliability_change", "is_null_action",
            )},
            "is_null_action": 1.0,
        }
    utility = UtilityModel()
    acceptance_model = AcceptanceModel()
    alternatives = routes[1:]
    alt_specs = [
        {
            "time": route.features.time,
            "capacity": float(metadata["alternative_capacities"][index]),
            "overlap": float(metadata["route_overlap"]) * (1.0 + 0.25 * index),
            "variability": route.features.variability,
            "bottleneck_load": route.features.time / max(float(metadata["alternative_capacities"][index]), 1.0),
            "legal": True,
        }
        for index, route in enumerate(alternatives)
    ]
    weights = allocation_weights(str(action["route_allocation"]), alt_specs)
    records = []
    route_options: dict[str, list[dict]] = {}
    for index, user in enumerate(users):
        vehicle_id = f"v{index:04d}"
        if vehicle_id not in active_ids:
            continue
        utilities = utility.utilities(user.preferences, routes)
        slack = preference_slack(utilities)
        feasible = []
        for alt_index, route in enumerate(alternatives):
            route_slack = float(slack[route.route_id])
            if route_slack > user.epsilon + 1e-12:
                continue
            time_saving = routes[0].features.time - route.features.time
            probability = acceptance_model.probability(
                route_slack,
                utilities[route.route_id] - utilities["main"],
                time_saving,
                routes[0].features.variability - route.features.variability,
                max(0.0, time_saving),
            )
            probability = float(np.clip(probability * float(condition["acceptance_multiplier"]), 0.0, 1.0))
            feasible.append({
                "route_index": alt_index,
                "preference_slack": route_slack,
                "acceptance_probability": probability,
                "marginal_network_benefit": time_saving * probability,
                "safety_exposure": route.features.risk + 0.01 * float(metadata["perturbation_strength"]),
            })
        if not feasible:
            continue
        preferred = max(feasible, key=lambda value: (value["marginal_network_benefit"], value["acceptance_probability"]))
        route_options[vehicle_id] = feasible
        preference = user.preferences.normalized()
        records.append({
            "vehicle_id": vehicle_id,
            "preference_slack": preferred["preference_slack"],
            "epsilon": float(user.epsilon),
            "acceptance_probability": preferred["acceptance_probability"],
            "marginal_network_benefit": preferred["marginal_network_benefit"],
            "safety_exposure": preferred["safety_exposure"],
            "weight_time": float(preference.time),
            "weight_reliability": float(preference.variability),
            "weight_safety": float(preference.risk),
        })
    ranked = rank_feasible_users(records, str(action["user_strategy"]))
    requested = len(active_ids) if float(action["reroute_fraction"]) >= 1.0 else int(round(float(action["reroute_fraction"]) * len(active_ids)))
    selected = ranked[:requested]
    assignments = {}
    selected_slack = []
    selected_acceptance = []
    selected_route_indices = []
    for record in selected:
        vehicle_id = str(record["vehicle_id"])
        available = route_options[vehicle_id]
        desired = deterministic_route_index(vehicle_id, weights)
        option = next((value for value in available if value["route_index"] == desired), None)
        if option is None:
            option = max(available, key=lambda value: value["marginal_network_benefit"])
        assignments[vehicle_id] = int(option["route_index"])
        selected_slack.append(float(option["preference_slack"]))
        selected_acceptance.append(float(option["acceptance_probability"]))
        selected_route_indices.append(int(option["route_index"]))
    expected = float(sum(selected_acceptance))
    expected_flow = expected * 3600.0 / max(generation_seconds, 1)
    realized_weights = [
        selected_route_indices.count(index) / max(len(selected_route_indices), 1)
        for index in range(len(alternatives))
    ]
    if not selected_route_indices:
        realized_weights = [0.0 for _ in alternatives]
    maximum_target = expected_flow * max(realized_weights, default=0.0)
    capacities = [float(value) for value in metadata["alternative_capacities"]]
    capacity_slack = min(
        (capacity - expected_flow * realized_weights[index]) / max(capacity, 1.0)
        for index, capacity in enumerate(capacities)
    ) if selected_route_indices else 1.0
    topology_factor = 1.5 if str(condition["topology"]) in {"merge", "signalized"} else 1.0
    average_length_ratio = float(metadata["route_length_ratio"])
    return {
        "selected_user_ids": [str(row["vehicle_id"]) for row in selected],
        "assignments": assignments,
        "route_weights": realized_weights,
        "proposed_rerouted_user_count": float(len(selected)),
        "expected_accepted_user_count": expected,
        "expected_rerouted_flow": expected_flow,
        "maximum_target_edge_load_increase": maximum_target,
        "maximum_source_edge_load_reduction": expected_flow,
        "bottleneck_flow_delta": maximum_target - 0.5 * expected_flow,
        "destination_capacity_slack": capacity_slack,
        "route_overlap_delta": expected_flow * float(metadata["route_overlap"]) / max(float(condition["demand"]), 1.0),
        "route_entropy": route_entropy(realized_weights) if selected_route_indices else 0.0,
        "affected_junction_count": 2.0 if selected_route_indices else 0.0,
        "expected_lane_change_demand_delta": expected_flow / max(float(condition["demand"]), 1.0),
        "conflict_zone_exposure_delta": topology_factor * maximum_target / max(sum(capacities), 1.0),
        "action_concentration_index": action_concentration_index(realized_weights) if selected_route_indices else 0.0,
        "average_preference_slack_selected": float(np.mean(selected_slack)) if selected_slack else 0.0,
        "p90_preference_slack_selected": float(np.percentile(selected_slack, 90)) if selected_slack else 0.0,
        "expected_acceptance_probability": float(np.mean(selected_acceptance)) if selected_acceptance else 0.0,
        "route_length_increase": (average_length_ratio - 1.0) * expected / max(len(active_ids), 1),
        "reliability_change": float(np.mean([alternatives[index].features.variability - routes[0].features.variability for index in selected_route_indices])) if selected_route_indices else 0.0,
        "is_null_action": 0.0,
    }


def run_v9_action(
    network: Path,
    metadata: Mapping[str, object],
    config: Mapping[str, object],
    condition: Mapping[str, object],
    predecision_seed: int,
    future_seed: int,
    action: AdaptiveAction | Mapping[str, object],
    analytical: Mapping[str, float],
    *,
    plan_actions: Sequence[AdaptiveAction | Mapping[str, object]] | None = None,
) -> dict:
    action_value = _action_dict(action)
    with tempfile.TemporaryDirectory(prefix="concordia-v9-run-") as temporary:
        directory = Path(temporary)
        run_config, route_file, expected = _write_run_files(
            directory, network, predecision_seed, int(condition["demand"]), int(config["vehicle_generation_seconds"])
        )
        binary = SumoAdapter.resolve_binary("sumo")
        if binary is None:
            raise RuntimeError("SUMO is unavailable")
        adapter = SumoAdapter(str(run_config), binary=binary)
        users = generate_population(expected, "s", "d", str(condition["heterogeneity"]), float(config["preference_epsilon"]), 5.0, predecision_seed)
        user_by_vehicle = {f"v{index:04d}": user for index, user in enumerate(users)}
        active_ids = {
            f"v{index:04d}" for index in range(expected)
            if ((index * 2654435761 + predecision_seed) % 10_000) < int(float(condition["penetration"]) * 10_000)
        }
        utility = UtilityModel()
        acceptance_model = AcceptanceModel()
        decision_time = float(config["decision_time_seconds"])
        temporal_window = float(config["temporal_window_seconds"])
        series = []
        frames = []
        departures = {}
        travel_times = []
        offers = accepted = rejected = diverted = 0
        maximum_regret = 0.0
        lane_changes = 0
        previous_lanes = {}
        planned = None
        candidate_plans = {}
        decision_routes = None
        adapter.start(predecision_seed)
        try:
            while adapter._traci.simulation.getMinExpectedNumber() > 0 and adapter._traci.simulation.getTime() < float(config["maximum_simulation_seconds"]):
                snapshot = adapter.step()
                now = float(snapshot.time)
                if decision_routes is None and now >= decision_time:
                    decision_routes = _route_features(adapter)
                    planned = _plan_action(action_value, users, active_ids, decision_routes, metadata, condition, int(config["vehicle_generation_seconds"]))
                    for item in plan_actions or ():
                        item_value = _action_dict(item)
                        candidate_plans[str(item_value["action_id"])] = _plan_action(
                            item_value, users, active_ids, decision_routes, metadata, condition, int(config["vehicle_generation_seconds"])
                        )
                for vehicle_id in adapter._traci.simulation.getDepartedIDList():
                    departures[vehicle_id] = now
                    if vehicle_id not in active_ids:
                        continue
                    routes = _route_features(adapter)
                    fastest = min(routes, key=lambda route: route.features.time)
                    if now <= decision_time or bool(action_value.get("is_null", False)):
                        if fastest.route_id != "main":
                            adapter._traci.vehicle.setRoute(vehicle_id, list(fastest.nodes))
                            diverted += 1
                        continue
                    if planned is None or vehicle_id not in planned["assignments"]:
                        if fastest.route_id != "main":
                            adapter._traci.vehicle.setRoute(vehicle_id, list(fastest.nodes))
                            diverted += 1
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
                    probability = float(np.clip(probability * float(condition["acceptance_multiplier"]), 0.0, 1.0))
                    latent_draw = int(
                        hashlib.sha256(f"{future_seed}::{vehicle_id}".encode()).hexdigest()[:13],
                        16,
                    ) / float(16**13)
                    if latent_draw <= probability:
                        adapter._traci.vehicle.setRoute(vehicle_id, list(target.nodes))
                        maximum_regret = max(maximum_regret, slack)
                        accepted += 1
                        diverted += 1
                    else:
                        rejected += 1
                for vehicle_id in adapter._traci.simulation.getArrivedIDList():
                    if vehicle_id in departures:
                        travel_times.append(now - departures[vehicle_id])
                headways, closing, drac, accelerations = [], [], [], []
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
                    leader_speed = max(0.0, float(adapter._traci.vehicle.getSpeed(leader[0]))) if leader else None
                    if gap is not None and leader_speed is not None:
                        headways.append(gap / max(speed, 0.1))
                        closing_speed = max(0.0, speed - leader_speed)
                        closing.append(closing_speed)
                        drac.append(closing_speed**2 / (2.0 * gap))
                    frames.append(TrajectoryFrame(now, vehicle_id, leader[0] if leader else None, gap, speed, leader_speed, accel))
                if decision_time - temporal_window < now <= decision_time:
                    selected = [snapshot.edges[edge] for edge in ALL_INTERNAL_EDGES]
                    main_speed = float(np.mean([snapshot.edges[edge].mean_speed_meters_per_second for edge in MAIN_EDGES[1:-1]]))
                    alternate_speed = float(np.mean([snapshot.edges[edge].mean_speed_meters_per_second for edge in (*ALT1_EDGES[1:-1], *ALT2_EDGES[1:-1])]))
                    halting = sum(int(adapter._traci.edge.getLastStepHaltingNumber(edge)) for edge in ALL_INTERNAL_EDGES)
                    vehicle_count = sum(int(adapter._traci.edge.getLastStepVehicleNumber(edge)) for edge in ALL_INTERNAL_EDGES)
                    lane_occupancies = [
                        float(adapter._traci.lane.getLastStepOccupancy(f"{edge}_{lane_index}"))
                        for edge in ALL_INTERNAL_EDGES
                        for lane_index in range(adapter._traci.edge.getLaneNumber(edge))
                    ]
                    series.append({
                        "time": now,
                        "density": float(np.mean([item.density_vehicles_per_km_per_lane for item in selected])),
                        "flow": float(np.mean([item.flow_vehicles_per_hour_per_lane for item in selected])),
                        "occupancy": float(np.mean([item.occupancy_percent or 0.0 for item in selected])),
                        "mean_speed": float(np.mean([item.mean_speed_meters_per_second for item in selected])),
                        "acceleration_variance": float(np.var(accelerations)) if accelerations else 0.0,
                        "queue_length": float(halting), "halting_count": float(halting),
                        "lane_occupancy": float(np.mean(lane_occupancies)) if lane_occupancies else 0.0,
                        "headway_mean": float(np.mean(headways)) if headways else 150.0,
                        "headway_variance": float(np.var(headways)) if headways else 0.0,
                        "minimum_headway": min(headways, default=150.0),
                        "closing_speed_p90": float(np.percentile(closing, 90)) if closing else 0.0,
                        "drac_proxy_p95": float(np.percentile(drac, 95)) if drac else 0.0,
                        "lane_change_density": lane_changes / max(1, vehicle_count),
                        "merge_interaction_density": sum(int(adapter._traci.edge.getLastStepVehicleNumber(edge)) for edge in ("m3", "a2", "b2", "out")) / 4.0,
                        "speed_differential": abs(main_speed - alternate_speed),
                        "hard_braking_recent_rate": sum(value < -3.0 for value in accelerations) / max(1, len(accelerations)),
                    })
        finally:
            adapter.close()
        if not series or planned is None:
            raise RuntimeError("v9 decision state was not observed")
        selected_slacks = [float(planned["average_preference_slack_selected"])] if planned["selected_user_ids"] else [0.0]
        preference_weights = [float(np.var(list(asdict(user.preferences).values()))) for user in users]
        preference = {
            "predicted_acceptance": float(planned["expected_acceptance_probability"]),
            "preference_slack_mean": float(np.mean(selected_slacks)),
            "preference_slack_std": float(np.std(selected_slacks)),
            "preference_variance": float(np.mean(preference_weights)) if preference_weights else 0.0,
        }
        features = _aggregate_features(series, condition, metadata, preference, analytical)
        safety = asdict(summarize_safety(frames))
        for name in ("ttc_values", "drac_values", "pet_values"):
            safety.pop(name)
        return {
            "policy": str(action_value["action_id"]),
            "action": action_value,
            "action_plan": planned,
            "candidate_plans": candidate_plans,
            "seed": predecision_seed,
            "future_seed": future_seed,
            "condition": dict(condition),
            "decision_time": decision_time,
            "feature_observation_end_time": max(row["time"] for row in series),
            "features_pre_decision": features,
            "predecision_series": series,
            "generated_vehicle_count": expected,
            "arrived_vehicle_count": len(travel_times),
            "censored_vehicle_count": expected - len(travel_times),
            "total_travel_time_seconds": float(np.sum(travel_times)),
            "mean_travel_time_seconds": float(np.mean(travel_times)) if travel_times else None,
            "navigated_vehicle_count": len(active_ids),
            "offer_count": offers,
            "accepted_count": accepted,
            "rejected_count": rejected,
            "acceptance_rate": accepted / max(1, offers),
            "diverted_vehicle_count": diverted,
            "maximum_affected_regret": maximum_regret,
            "all_executed_routes_legal": bool(metadata["all_routes_legal"]),
            "safety": safety,
            "network_hash": _sha(network),
            "route_file_hash": _sha(route_file),
        }
