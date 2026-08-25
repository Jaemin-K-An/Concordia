#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/concordia-matplotlib-v2")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/concordia-cache-v2")

import matplotlib
import networkx as nx
import numpy as np
import sumolib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from concordia.behavior import AcceptanceModel
from concordia.evaluation import ExperimentRegistry, capture_source_state, paired_comparison
from concordia.models import Route, RouteFeatures
from concordia.populations import generate_population
from concordia.preferences import UtilityModel, preference_slack
from concordia.safety import TrajectoryFrame, summarize_safety
from concordia.simulation import SumoAdapter


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiments" / "real_topology_policy_matrix.yaml"
OUTPUT = ROOT / "artifacts" / "studies" / "real_topology_policy_matrix"


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_network(directory: Path, config: dict) -> Path:
    source = ROOT / config["source_osm"]
    network = directory / "gangnam.net.xml"
    netconvert = SumoAdapter.resolve_binary("netconvert")
    if netconvert is None:
        raise RuntimeError("netconvert is unavailable")
    completed = subprocess.run(
        [
            netconvert,
            "--osm-files",
            str(source),
            "--output-file",
            str(network),
            "--geometry.remove",
            "true",
            "--ramps.guess",
            "true",
            "--junctions.join",
            "true",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode or not network.is_file():
        raise RuntimeError(f"real OSM conversion failed: {completed.stderr[-1000:]}")
    return network


def _legal_routes(network_path: Path, origin: str, destination: str, count: int = 3):
    network = sumolib.net.readNet(str(network_path))
    graph = nx.DiGraph()
    edge_for_pair = {}
    for edge in network.getEdges(withInternal=False):
        if not edge.allows("passenger"):
            continue
        source = edge.getFromNode().getID()
        target = edge.getToNode().getID()
        weight = edge.getLength() / max(edge.getSpeed(), 1e-6)
        if (source, target) not in edge_for_pair or weight < graph[source][target]["weight"]:
            graph.add_edge(source, target, weight=weight)
            edge_for_pair[(source, target)] = edge.getID()
    paths = []
    for nodes in nx.shortest_simple_paths(graph, origin, destination, weight="weight"):
        edge_ids = tuple(edge_for_pair[pair] for pair in zip(nodes, nodes[1:]))
        paths.append((tuple(nodes), edge_ids))
        if len(paths) >= count:
            break
    if len(paths) < 2:
        raise RuntimeError("real topology has fewer than two legal SUMO route alternatives")
    return network, paths


def _write_config(directory: Path, network: Path, paths) -> Path:
    route_path = directory / "routes.rou.xml"
    route_lines = [
        f'  <route id="r{index}" edges="{" ".join(edges)}"/>'
        for index, (_, edges) in enumerate(paths)
    ]
    route_path.write_text(
        "<routes>\n"
        '  <vType id="passenger" vClass="passenger" accel="2.6" decel="4.5" '
        'emergencyDecel="8.0" sigma="0.3" tau="1.0"/>\n'
        + "\n".join(route_lines)
        + "\n</routes>\n",
        encoding="utf-8",
    )
    run_config = directory / "run.sumocfg"
    run_config.write_text(
        f"""<configuration>
  <input><net-file value="{network}"/><route-files value="{route_path}"/></input>
  <time><step-length value="1"/></time>
  <processing><collision.action value="warn"/><time-to-teleport value="300"/></processing>
  <report><no-step-log value="true"/><duration-log.disable value="true"/></report>
</configuration>\n""",
        encoding="utf-8",
    )
    return run_config


def _features(adapter: SumoAdapter, paths) -> dict[str, Route]:
    routes = {}
    for index, (nodes, edges) in enumerate(paths):
        eta = sum(float(adapter._traci.edge.getTraveltime(edge)) for edge in edges) / 60.0
        tradeoffs = (
            {"variability": 3.0, "risk": 0.08, "complexity": 0.60, "familiarity": 1.0},
            {"variability": 1.0, "risk": 0.03, "complexity": 0.35, "familiarity": 0.4},
            {"variability": 0.5, "risk": 0.02, "complexity": 0.20, "familiarity": 0.2},
        )[min(index, 2)]
        routes[f"r{index}"] = Route(
            f"r{index}",
            nodes,
            RouteFeatures(
                eta,
                variability=tradeoffs["variability"],
                risk=tradeoffs["risk"],
                complexity=tradeoffs["complexity"],
                familiarity=tradeoffs["familiarity"],
            ),
        )
    return routes


def _run_one(
    run_config: Path,
    network,
    paths,
    config: dict,
    mode: str,
    policy: str,
    seed: int,
) -> dict:
    started = time.perf_counter()
    binary = SumoAdapter.resolve_binary("sumo")
    if binary is None:
        raise RuntimeError("SUMO is unavailable")
    adapter = SumoAdapter(str(run_config), binary=binary)
    demand = int(config["demand_vehicles_per_hour"])
    spacing = 3600.0 / demand
    count = int(math.floor(int(config["vehicle_generation_seconds"]) / spacing))
    epsilon = float(config["tests"][mode]["preference_epsilon"])
    users = generate_population(count, "origin", "destination", "high", epsilon, 5.0, seed)
    rng = random.Random(seed * 101 + (0 if policy == "B1" else 1))
    utility_model = UtilityModel()
    acceptance_model = AcceptanceModel()
    departures = {}
    travel_times = []
    frames = []
    assignments = Counter()
    recommendation_load = Counter()
    regrets = []
    offers = accepted = rejected = 0
    edge_accumulator = defaultdict(
        lambda: {"flow": [], "speed": [], "phantom": [], "safety": []}
    )
    adapter.start(seed)
    next_vehicle = 0
    try:
        while adapter._traci.simulation.getTime() < float(config["maximum_simulation_seconds"]):
            now = float(adapter._traci.simulation.getTime())
            while next_vehicle < count and next_vehicle * spacing <= now + 1e-9:
                routes = _features(adapter, paths)
                user = users[next_vehicle]
                utilities = utility_model.utilities(user.preferences, routes.values())
                slacks = preference_slack(utilities)
                private_id = max(sorted(utilities), key=utilities.__getitem__)
                fastest_id = min(routes, key=lambda route_id: routes[route_id].features.time)
                chosen_id = fastest_id if policy == "B1" else private_id
                if policy == "B6" and fastest_id != private_id:
                    offers += 1
                    selected = routes[fastest_id].features
                    current = routes[private_id].features
                    probability = acceptance_model.probability(
                        slacks[fastest_id],
                        utilities[fastest_id] - utilities[private_id],
                        current.time - selected.time,
                        current.variability - selected.variability,
                        max(0.0, current.time - selected.time),
                    )
                    if slacks[fastest_id] <= epsilon and rng.random() <= probability:
                        chosen_id = fastest_id
                        accepted += 1
                    else:
                        rejected += 1
                regrets.append(slacks[chosen_id])
                vehicle_id = f"v{next_vehicle:04d}"
                adapter._traci.vehicle.add(
                    vehicle_id,
                    chosen_id,
                    typeID="passenger",
                    depart="now",
                )
                departures[vehicle_id] = now
                assignments[chosen_id] += 1
                if chosen_id != private_id:
                    for edge in paths[int(chosen_id[1:])][1]:
                        recommendation_load[edge] += 1
                next_vehicle += 1
            adapter._traci.simulationStep()
            now = float(adapter._traci.simulation.getTime())
            for vehicle_id in adapter._traci.simulation.getArrivedIDList():
                if vehicle_id in departures:
                    travel_times.append(now - departures[vehicle_id])
            for vehicle_id in adapter._traci.vehicle.getIDList():
                leader = adapter._traci.vehicle.getLeader(vehicle_id, 150.0)
                frames.append(
                    TrajectoryFrame(
                        time=now,
                        follower_id=vehicle_id,
                        leader_id=leader[0] if leader else None,
                        gap=max(1e-6, float(leader[1])) if leader else None,
                        follower_speed=max(
                            0.0, float(adapter._traci.vehicle.getSpeed(vehicle_id))
                        ),
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
            if round(now) % 5 == 0:
                for edge in network.getEdges(withInternal=False):
                    edge_id = edge.getID()
                    vehicle_ids = tuple(
                        adapter._traci.edge.getLastStepVehicleIDs(edge_id)
                    )
                    speeds = [
                        max(0.0, float(adapter._traci.vehicle.getSpeed(item)))
                        for item in vehicle_ids
                    ]
                    accelerations = [
                        float(adapter._traci.vehicle.getAcceleration(item))
                        for item in vehicle_ids
                    ]
                    mean_speed = float(np.mean(speeds)) if speeds else edge.getSpeed()
                    speed_cv = (
                        float(np.std(speeds) / np.mean(speeds))
                        if len(speeds) > 1 and np.mean(speeds) > 1e-12
                        else 0.0
                    )
                    lane_kilometers = (
                        edge.getLaneNumber() * edge.getLength() / 1000.0
                    )
                    density = len(vehicle_ids) / max(lane_kilometers, 1e-9)
                    flow = density * mean_speed * 3.6
                    values = edge_accumulator[edge_id]
                    values["flow"].append(flow)
                    values["speed"].append(mean_speed)
                    saturation = flow / 1800.0
                    values["phantom"].append(
                        min(1.0, 0.5 * saturation + 0.5 * speed_cv)
                    )
                    values["safety"].append(
                        float(np.var(accelerations)) if len(accelerations) > 1 else 0.0
                    )
            if next_vehicle >= count and adapter._traci.simulation.getMinExpectedNumber() <= 0:
                break
    finally:
        adapter.close()
    safety = summarize_safety(frames)
    safety_dict = asdict(safety)
    safety_dict.pop("ttc_values")
    safety_dict.pop("drac_values")
    safety_dict.pop("pet_values")
    total = sum(assignments.values())
    shares = [value / total for value in assignments.values()]
    entropy = -sum(share * math.log(share) for share in shares if share > 0)
    edge_metrics = {
        edge: {
            "flow": float(np.mean(values["flow"])),
            "speed": float(np.mean(values["speed"])),
            "phantom_risk": float(np.mean(values["phantom"])),
            "safety_risk": float(np.mean(values["safety"])),
            "recommendation_load": recommendation_load[edge],
        }
        for edge, values in edge_accumulator.items()
    }
    return {
        "run_id": f"{mode}-{policy}-{seed}",
        "mode": mode,
        "policy": policy,
        "seed": seed,
        "demand_provenance": config["demand_provenance"],
        "generated_vehicle_count": count,
        "arrived_vehicle_count": len(travel_times),
        "censored_vehicle_count": count - len(travel_times),
        "total_travel_time_seconds": float(np.sum(travel_times)),
        "mean_travel_time_seconds": float(np.mean(travel_times)) if travel_times else None,
        "mean_regret": float(np.mean(regrets)),
        "p95_regret": float(np.percentile(regrets, 95)),
        "max_regret": float(np.max(regrets)),
        "offer_count": offers,
        "accepted_count": accepted,
        "rejected_count": rejected,
        "acceptance_rate": accepted / max(1, offers),
        "route_entropy": entropy,
        "route_concentration_hhi": sum(share**2 for share in shares),
        "route_counts": dict(assignments),
        "safety": safety_dict,
        "runtime_seconds": time.perf_counter() - started,
        "edge_metrics": edge_metrics,
    }


def _statistics(rows: list[dict]) -> dict:
    pairs = defaultdict(dict)
    for row in rows:
        pairs[(row["mode"], row["seed"])][row["policy"]] = row
    output = {}
    for mode in sorted({row["mode"] for row in rows}):
        selected = [value for (pair_mode, _), value in pairs.items() if pair_mode == mode]
        b1 = [pair["B1"]["total_travel_time_seconds"] for pair in selected]
        b6 = [pair["B6"]["total_travel_time_seconds"] for pair in selected]
        output[mode] = {
            "TTT_paired": paired_comparison(b1, b6, seed=59),
            "B1_mean_TTT_seconds": float(np.mean(b1)),
            "B6_mean_TTT_seconds": float(np.mean(b6)),
            "B6_mean_regret": float(np.mean([pair["B6"]["mean_regret"] for pair in selected])),
            "B6_acceptance_rate": float(
                np.mean([pair["B6"]["acceptance_rate"] for pair in selected])
            ),
        }
    return output


def _qgis_layer(network, rows: list[dict], destination: Path) -> None:
    by_policy = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["mode"] != "transfer":
            continue
        for edge, values in row["edge_metrics"].items():
            by_policy[row["policy"]][edge].append(values)
    features = []
    # The checked-in SUMO runtime does not depend on pyproj.  netconvert stores
    # the UTM-to-network offset even when sumolib cannot invert it to WGS84, so
    # export a truthful projected GeoJSON layer instead of guessing lon/lat.
    # The Gangnam source is UTM zone 52N (EPSG:32652); QGIS reads this explicit
    # CRS and the coordinates remain metre-valued and spatially exact.
    offset_x, offset_y = network.getLocationOffset()
    for edge in network.getEdges(withInternal=False):
        edge_id = edge.getID()
        baseline_values = by_policy["B1"].get(edge_id, [])
        adaptive_values = by_policy["B6"].get(edge_id, [])

        def mean(values, field):
            return float(np.mean([item[field] for item in values])) if values else 0.0

        baseline_flow = mean(baseline_values, "flow")
        adaptive_flow = mean(adaptive_values, "flow")
        baseline_speed = mean(baseline_values, "speed")
        adaptive_speed = mean(adaptive_values, "speed")
        shape = [
            [float(x - offset_x), float(y - offset_y)]
            for x, y in edge.getShape()
        ]
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": shape},
                "properties": {
                    "edge_id": edge_id,
                    "baseline_flow": baseline_flow,
                    "adaptive_flow": adaptive_flow,
                    "delta_flow": adaptive_flow - baseline_flow,
                    "baseline_speed": baseline_speed,
                    "adaptive_speed": adaptive_speed,
                    "phantom_risk": mean(adaptive_values, "phantom_risk"),
                    "safety_risk": mean(adaptive_values, "safety_risk"),
                    "recommendation_load": mean(adaptive_values, "recommendation_load"),
                    "bottleneck_score": max(
                        0.0,
                        adaptive_flow / 1800.0
                        + max(0.0, 1.0 - adaptive_speed / max(edge.getSpeed(), 1e-6)),
                    ),
                },
            }
        )
    _write_json(
        destination,
        {
            "type": "FeatureCollection",
            "name": "gangnam_real_topology_policy_delta",
            "crs": {
                "type": "name",
                "properties": {"name": "urn:ogc:def:crs:EPSG::32652"},
            },
            "coordinate_units": "metres",
            "demand_provenance": "synthetic demand on real topology",
            "features": features,
        },
    )


def _figures(rows: list[dict], layer_path: Path) -> list[Path]:
    directory = OUTPUT / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    outputs = []
    fig, axis = plt.subplots(figsize=(6.2, 4.2))
    values = [
        [row["total_travel_time_seconds"] for row in rows if row["policy"] == policy]
        for policy in ("B1", "B6")
    ]
    axis.boxplot(values, tick_labels=["B1", "B6"])
    axis.set_ylabel("Network TTT (s)")
    axis.set_title("Real topology, synthetic demand")
    fig.tight_layout()
    path = directory / "real_topology_ttt_comparison.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)

    layer = json.loads(layer_path.read_text(encoding="utf-8"))
    fig, axis = plt.subplots(figsize=(7.2, 7.2))
    for feature in layer["features"]:
        coordinates = feature["geometry"]["coordinates"]
        if not coordinates:
            continue
        x, y = zip(*coordinates)
        delta = feature["properties"]["delta_flow"]
        color = "#111111" if delta > 0 else "#aaaaaa"
        width = 0.3 + min(2.5, abs(delta) / 100.0)
        axis.plot(x, y, color=color, linewidth=width, alpha=0.8)
    axis.set_aspect("equal")
    axis.set_axis_off()
    axis.set_title("Recommendation load redistribution (synthetic demand)")
    fig.tight_layout()
    path = directory / "real_topology_recommendation_map.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)
    return outputs


def run() -> Path:
    source_commit, source_dirty = capture_source_state()
    started = datetime.now(timezone.utc)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / config["source_manifest"]).read_text(encoding="utf-8"))
    source_path = ROOT / config["source_osm"]
    if _sha(source_path) != manifest["checksum_sha256"]:
        raise RuntimeError("real OSM source checksum does not match provenance manifest")
    with tempfile.TemporaryDirectory(prefix="concordia-real-policy-") as temporary:
        directory = Path(temporary)
        network_path = _build_network(directory, config)
        network, paths = _legal_routes(
            network_path,
            config["origin_osm_node"],
            config["destination_osm_node"],
        )
        run_config = _write_config(directory, network_path, paths)
        rows = []
        for mode in config["tests"]:
            for seed in config["seeds"]:
                for policy in config["policies"]:
                    rows.append(
                        _run_one(
                            run_config,
                            network,
                            paths,
                            config,
                            mode,
                            policy,
                            int(seed),
                        )
                    )
        network_hash = _sha(network_path)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    raw_path = OUTPUT / "raw_metrics.json"
    statistics_path = OUTPUT / "statistical_tests.json"
    layer_path = OUTPUT / "gangnam_policy_delta.geojson"
    _write_json(raw_path, rows)
    statistics = _statistics(rows)
    _write_json(statistics_path, statistics)
    _qgis_layer(network, rows, layer_path)
    figures = _figures(rows, layer_path)
    summary = {
        "complete": True,
        "study": "Study III — Real Topology Transfer",
        "run_count": len(rows),
        "legal_route_count": len(paths),
        "source_osm_sha256": manifest["checksum_sha256"],
        "sumo_network_sha256": network_hash,
        "demand_provenance": config["demand_provenance"],
        "transfer_and_calibrated_separated": True,
        "statistics": statistics,
        "claim_boundary": config["claim_boundary"],
    }
    summary_path = OUTPUT / "summary.json"
    _write_json(summary_path, summary)
    ended = datetime.now(timezone.utc)
    outputs = [raw_path, statistics_path, layer_path, summary_path, *figures]
    run_dir = ExperimentRegistry(str(ROOT / "artifacts" / "runs")).create(
        config,
        summary,
        simulator_version=SumoAdapter.simulator_version(),
        input_paths=(
            str(CONFIG.relative_to(ROOT)),
            config["source_osm"],
            config["source_manifest"],
        ),
        external_output_paths=tuple(str(path.relative_to(ROOT)) for path in outputs),
        started_at=started,
        ended_at=ended,
        source_commit=source_commit,
        source_dirty=source_dirty,
    )
    shutil.copyfile(run_dir / "manifest.json", OUTPUT / "manifest.json")
    print(summary_path)
    return summary_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-if-valid", action="store_true")
    arguments = parser.parse_args()
    existing = OUTPUT / "summary.json"
    if arguments.reuse_if_valid and existing.is_file() and json.loads(
        existing.read_text(encoding="utf-8")
    ).get("complete"):
        print(existing)
    else:
        run()
