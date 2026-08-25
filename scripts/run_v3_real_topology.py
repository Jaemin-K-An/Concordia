#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/concordia-matplotlib-v3")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/concordia-cache-v3")

import matplotlib
import networkx as nx
import numpy as np
import sumolib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from concordia.behavior import AcceptanceModel
from concordia.evaluation import ExperimentRegistry, capture_source_state, summarize_selective_policy
from concordia.feasibility import BootstrapFeasibilityEnsemble, FEATURE_SCHEMA, FeasibilityGate
from concordia.feasibility.dataset import build_alignment_case
from concordia.models import Route, RouteFeatures
from concordia.populations import generate_population
from concordia.preferences import UtilityModel, preference_slack
from concordia.selective import SelectiveInterventionPolicy
from concordia.simulation import SumoAdapter
from run_real_topology_study import (
    _build_network,
    _paths_are_legal,
    _qgis_layer,
    _run_one,
    _write_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/v3/real_topology.yaml"
MODEL_STUDY = ROOT / "artifacts/studies/v3_feasibility_prediction"
OUTPUT = ROOT / "artifacts/studies/v3_real_topology_selective"


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _passenger_graph(network) -> tuple[nx.DiGraph, dict[tuple[str, str], str]]:
    graph = nx.DiGraph()
    edge_for_pair = {}
    for edge in network.getEdges(withInternal=False):
        if not edge.allows("passenger"):
            continue
        source = edge.getFromNode().getID()
        target = edge.getToNode().getID()
        weight = edge.getLength() / max(edge.getSpeed(), 1e-9)
        if not graph.has_edge(source, target) or weight < graph[source][target]["weight"]:
            graph.add_edge(source, target, weight=weight)
            edge_for_pair[(source, target)] = edge.getID()
    return graph, edge_for_pair


def _select_unseen_od(network_path: Path, development_od: list[str]):
    network = sumolib.net.readNet(str(network_path))
    graph, edge_for_pair = _passenger_graph(network)
    development_origin, development_destination = development_od
    if development_origin not in graph or development_destination not in graph:
        raise RuntimeError("development OD is absent from converted passenger graph")
    primary = nx.shortest_path(
        graph, development_origin, development_destination, weight="weight"
    )
    candidates = []
    for left_index in range(min(4, len(primary) - 2)):
        for right_index in range(max(left_index + 2, len(primary) - 4), len(primary)):
            candidates.append((primary[left_index], primary[right_index]))
    nodes = sorted(graph)
    stride = max(1, len(nodes) // 20)
    candidates.extend((origin, destination) for origin in nodes[::stride] for destination in nodes[::stride])
    seen = set()
    best = None
    for origin, destination in candidates:
        if origin == destination or [origin, destination] == development_od or (origin, destination) in seen:
            continue
        seen.add((origin, destination))
        try:
            paths = []
            for node_path in nx.shortest_simple_paths(graph, origin, destination, weight="weight"):
                edge_ids = tuple(edge_for_pair[pair] for pair in zip(node_path, node_path[1:]))
                paths.append((tuple(node_path), edge_ids))
                if len(paths) == 3:
                    break
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        if len(paths) < 2:
            continue
        score = (len(paths), min(len(path[1]) for path in paths))
        if best is None or score > best[0]:
            best = (score, origin, destination, paths)
        if score[0] == 3 and score[1] >= 5:
            break
    if best is None:
        raise RuntimeError("could not locate an unseen connected OD with alternatives")
    _, origin, destination, paths = best
    return network, origin, destination, paths


def _route_overlap(paths) -> tuple[float, float]:
    edge_sets = [set(edges) for _nodes, edges in paths]
    values = [
        len(left & right) / max(1, len(left | right))
        for index, left in enumerate(edge_sets)
        for right in edge_sets[index + 1 :]
    ]
    return (float(np.mean(values)) if values else 0.0, float(np.max(values)) if values else 0.0)


def _actual_alignment_features(network, paths, seed: int, demand: int, frozen: dict) -> dict:
    case = build_alignment_case(
        scenario="ring",
        seed=seed,
        demand_scale=demand / 600.0,
        heterogeneity="high",
        navigation_penetration=1.0,
        user_count=6,
        regret_limit=0.08,
        epsilon_grid=[0.0, 0.02, 0.04, 0.08, 0.12, 0.16],
        minimum_relative_ttt_gain=float(frozen["minimum_relative_ttt_gain"]),
        safety_delta=float(frozen["safety_delta"]),
        source_split="real_topology_transfer",
    )
    features = dict(case["features"])
    route_objects = []
    path_capacities = []
    tradeoffs = (
        (3.0, 0.08, 0.60, 1.0),
        (1.0, 0.03, 0.35, 0.4),
        (0.5, 0.02, 0.20, 0.2),
    )
    for index, (nodes, edge_ids) in enumerate(paths):
        edges = [network.getEdge(edge_id) for edge_id in edge_ids]
        eta = sum(edge.getLength() / max(edge.getSpeed(), 1e-9) for edge in edges) / 60.0
        variability, risk, complexity, familiarity = tradeoffs[min(index, 2)]
        route_objects.append(
            Route(
                f"r{index}",
                nodes,
                RouteFeatures(eta, variability, 0.0, risk, complexity, familiarity),
            )
        )
        path_capacities.append(min(edge.getLaneNumber() * 1800.0 for edge in edges))
    users = generate_population(100, "origin", "destination", "high", 0.08, 5.0, seed)
    utility_model = UtilityModel()
    acceptance_model = AcceptanceModel()
    fastest = min(route_objects, key=lambda route: route.features.time)
    opportunity_mass = 0.0
    opportunity_count = 0
    maximum_benefit = 0.0
    acceptance_values = []
    for user in users:
        utilities = utility_model.utilities(user.preferences, route_objects)
        slack = preference_slack(utilities)
        private = max(route_objects, key=lambda route: utilities[route.route_id])
        benefit = max(0.0, private.features.time - fastest.features.time)
        if fastest.route_id != private.route_id and slack[fastest.route_id] <= user.epsilon:
            probability = acceptance_model.probability(
                slack[fastest.route_id],
                utilities[fastest.route_id] - utilities[private.route_id],
                benefit,
                private.features.variability - fastest.features.variability,
                benefit,
            )
            weighted = benefit * demand / len(users) * probability
            opportunity_mass += weighted
            opportunity_count += 1
            maximum_benefit = max(maximum_benefit, weighted)
            acceptance_values.append(probability)
    overlap_mean, overlap_max = _route_overlap(paths)
    primary_capacity = path_capacities[0]
    alternative_capacity = sum(path_capacities[1:])
    all_edges = [edge for _nodes, edge_ids in paths for edge in edge_ids]
    most_shared = max(all_edges.count(edge) for edge in set(all_edges)) / len(paths)
    features.update(
        {
            "demand_scale": demand / 600.0,
            "volume_capacity_ratio": demand / max(primary_capacity, 1e-9),
            "route_overlap": overlap_mean,
            "edge_disjointness": 1.0 - overlap_max,
            "alternative_capacity_ratio": alternative_capacity / max(primary_capacity, 1e-9),
            "bottleneck_centrality": most_shared,
            "cut_sensitivity": overlap_max,
            "route_diversity": len(paths) * (1.0 - overlap_mean),
            "alignment_opportunity_mass": opportunity_mass,
            "alignment_potential_score": opportunity_mass / max(demand, 1e-9),
            "alignment_opportunity_count": float(opportunity_count),
            "maximum_single_benefit": maximum_benefit,
            "acceptance_probability": float(np.mean(acceptance_values)) if acceptance_values else 0.0,
            "phantom_risk": min(1.0, demand / max(primary_capacity, 1e-9) * (0.5 + overlap_mean)),
        }
    )
    features["heterogeneity_rad_interaction"] = features["preference_variance"] * features[
        "route_attribute_diversity"
    ]
    return features


def _regression_predict(package: dict, matrix: np.ndarray) -> np.ndarray:
    mean = np.asarray(package["mean"], dtype=float)
    scale = np.asarray(package["scale"], dtype=float)
    coefficients = np.asarray(package["coefficients"], dtype=float)
    return package["intercept"] + ((matrix - mean) / scale) @ coefficients


def _figures(rows: list[dict], metrics: dict, layer_path: Path) -> list[Path]:
    directory = OUTPUT / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    outputs = []
    fig, axis = plt.subplots(figsize=(6.2, 4.2))
    axis.boxplot(
        [[row["total_travel_time_seconds"] for row in rows if row["policy"] == policy] for policy in ("B1", "B6", "V3")],
        tick_labels=["B1", "B6", "V3"],
    )
    axis.set(ylabel="Network TTT (s)", title="Unseen OD on real OSM geometry")
    fig.tight_layout()
    path = directory / "real_topology_v3_ttt.png"
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
        axis.plot(x, y, color="#0b6e4f" if delta > 0 else "#aaaaaa", linewidth=0.4 + min(2.2, abs(delta) / 100.0))
    axis.set_aspect("equal")
    axis.set_axis_off()
    fig.tight_layout()
    path = directory / "real_topology_v3_map.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)
    fig, axis = plt.subplots(figsize=(6.2, 4.2))
    primary = metrics["V3"]
    axis.bar(
        ["precision", "coverage", "PBR"],
        [primary["intervention_precision"], primary["coverage"], primary["population_benefit_rate"]],
        color="#0b6e4f",
    )
    axis.set_ylim(0, 1)
    fig.tight_layout()
    path = directory / "real_topology_selectivity.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)
    return outputs


def run() -> Path:
    source_commit, source_dirty = capture_source_state()
    started = datetime.now(timezone.utc)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    source_path = ROOT / config["source_osm"]
    manifest = json.loads((ROOT / config["source_manifest"]).read_text(encoding="utf-8"))
    if _sha(source_path) != manifest["checksum_sha256"]:
        raise RuntimeError("real OSM source checksum does not match its provenance manifest")
    frozen_path = ROOT / "configs/v3/frozen_thresholds.yaml"
    selected_path = MODEL_STUDY / "selected_model.json"
    frozen = yaml.safe_load(frozen_path.read_text(encoding="utf-8"))
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    if _sha(selected_path) != frozen["selected_model_hash"]:
        raise RuntimeError("selected feasibility model changed after freeze")
    ensemble = BootstrapFeasibilityEnsemble.from_dict(selected["ensemble"])
    policy = SelectiveInterventionPolicy(
        FeasibilityGate(
            probability_threshold=float(frozen["p_win_threshold"]),
            maximum_uncertainty=float(frozen["maximum_uncertainty"]),
            safety_delta=float(frozen["safety_delta"]),
            minimum_acceptance_probability=float(frozen["minimum_acceptance_probability"]),
            minimum_ttt_lcb_gain=float(frozen["minimum_relative_ttt_gain"]),
            maximum_tail_loss=float(frozen["maximum_tail_loss"]),
        )
    )
    with tempfile.TemporaryDirectory(prefix="concordia-v3-real-") as temporary:
        directory = Path(temporary)
        network_path = _build_network(directory, config)
        network, origin, destination, paths = _select_unseen_od(network_path, config["development_od"])
        run_config = _write_config(directory, network_path, paths)
        actual_pairs = defaultdict(dict)
        for demand in config["demand_vehicles_per_hour"]:
            run_parameters = {
                **config,
                "demand_vehicles_per_hour": int(demand),
                "demand_provenance": "synthetic demand on unseen OD and real OSM geometry",
                "tests": {"unseen": {"preference_epsilon": 0.08}},
            }
            for seed in config["seeds"]:
                for policy_name in ("B1", "B6"):
                    row = _run_one(
                        run_config,
                        network,
                        paths,
                        run_parameters,
                        "unseen",
                        policy_name,
                        int(seed),
                    )
                    actual_pairs[(int(seed), int(demand))][policy_name] = row
        network_hash = _sha(network_path)
    decision_log = []
    policy_rows = {"B6": [], "V3": []}
    raw_rows = []
    qgis_rows = []
    for (seed, demand), pair in sorted(actual_pairs.items()):
        features = _actual_alignment_features(network, paths, seed, demand, frozen)
        matrix = np.asarray([[features[name] for name in FEATURE_SCHEMA]], dtype=float)
        probability, uncertainty, lower = ensemble.predict(matrix)
        benefit = _regression_predict(selected["benefit_regression"], matrix)[0]
        benefit_lcb = benefit - frozen["benefit_lcb_z"] * selected["benefit_regression"][
            "residual_standard_deviation"
        ]
        safety = _regression_predict(selected["safety_regression"], matrix)[0]
        safety_upper = safety + frozen["benefit_lcb_z"] * selected["safety_regression"][
            "residual_standard_deviation"
        ]
        decision = policy.decide(
            case_id=f"real-s{seed}-d{demand}",
            p_win=float(probability[0]),
            p_win_lower=float(lower[0]),
            uncertainty=float(uncertainty[0]),
            alignment_potential=features["alignment_potential_score"],
            route_overlap=features["route_overlap"],
            safety_upper_difference=float(safety_upper),
            acceptance_probability=features["acceptance_probability"],
            ttt_lcb_gain=float(benefit_lcb),
            predicted_tail_loss=max(0.0, -float(benefit_lcb)),
            legal=_paths_are_legal(network, paths),
        )
        b1, b6 = pair["B1"], pair["B6"]
        relative_gain = (b1["total_travel_time_seconds"] - b6["total_travel_time_seconds"]) / max(
            b1["total_travel_time_seconds"], 1e-9
        )
        safety_difference = b6["safety"]["cvar_drac_95"] - b1["safety"]["cvar_drac_95"]
        success = (
            relative_gain >= frozen["minimum_relative_ttt_gain"]
            and b6["max_regret"] <= 0.08 + 1e-10
            and safety_difference <= frozen["safety_delta"]
            and _paths_are_legal(network, paths)
        )
        selected_row = b6 if decision.intervene else b1
        v3_row = {**selected_row, "policy": "V3", "run_id": f"unseen-V3-{seed}-{demand}"}
        raw_rows.extend((b1, b6, v3_row))
        qgis_rows.extend(({**b1, "mode": "transfer"}, {**v3_row, "mode": "transfer", "policy": "B6"}))
        for policy_name, intervene in (("B6", True), ("V3", decision.intervene)):
            selected_ttt = b6["total_travel_time_seconds"] if intervene else b1["total_travel_time_seconds"]
            policy_rows[policy_name].append(
                {
                    "case_id": decision.case_id,
                    "intervene": intervene,
                    "success": bool(intervene and success),
                    "counterfactual_success": success,
                    "system_ttt_gain": b1["total_travel_time_seconds"] - selected_ttt,
                    "baseline_ttt": b1["total_travel_time_seconds"],
                    "selected_ttt": selected_ttt,
                    "regret_violation": b6["max_regret"] > 0.08 + 1e-10,
                    "safety_violation": safety_difference > frozen["safety_delta"],
                    "legal_violation": not _paths_are_legal(network, paths),
                }
            )
        decision_log.append(
            {
                **decision.__dict__,
                "outcome": "SUCCESS" if decision.intervene and success else "FAILURE" if decision.intervene else "ABSTAIN",
                "features": features,
                "relative_ttt_gain_realized": relative_gain,
                "safety_difference_realized": safety_difference,
                "threshold_hash": _sha(frozen_path),
            }
        )
    metrics = {name: summarize_selective_policy(values) for name, values in policy_rows.items()}
    statistics = {
        "B6": metrics["B6"],
        "V3": metrics["V3"],
        "H12_all_recommended_paths_legal": _paths_are_legal(network, paths),
        "H12_unseen_od": [origin, destination] != config["development_od"],
        "H12_positive_net_benefit": metrics["V3"]["mean_network_ttt_gain"] > 0,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    raw_path = OUTPUT / "raw_metrics.json"
    processed_path = OUTPUT / "processed_metrics.json"
    statistical_path = OUTPUT / "statistical_tests.json"
    summary_path = OUTPUT / "summary.json"
    decision_path = OUTPUT / "decision_log.json"
    layer_path = OUTPUT / "gangnam_v3_policy_delta.geojson"
    _write(raw_path, raw_rows)
    _write(processed_path, metrics)
    _write(statistical_path, statistics)
    _write(decision_path, decision_log)
    _qgis_layer(network, qgis_rows, layer_path)
    layer = json.loads(layer_path.read_text(encoding="utf-8"))
    layer["name"] = "gangnam_unseen_od_v3_policy_delta"
    layer["development_od"] = config["development_od"]
    layer["evaluation_od"] = [origin, destination]
    _write(layer_path, layer)
    overlap_mean, overlap_max = _route_overlap(paths)
    summary = {
        "complete": True,
        "study": "Study VIII — Real-Topology Selective Transfer",
        "simulator_version": SumoAdapter.simulator_version(),
        "source_osm_sha256": manifest["checksum_sha256"],
        "sumo_network_sha256": network_hash,
        "development_od": config["development_od"],
        "evaluation_od": [origin, destination],
        "evaluation_od_unseen": [origin, destination] != config["development_od"],
        "legal_route_count": len(paths),
        "all_recommended_paths_legal": _paths_are_legal(network, paths),
        "route_overlap_jaccard_mean": overlap_mean,
        "route_overlap_jaccard_max": overlap_max,
        "policy_metrics": metrics,
        "statistics": statistics,
        "claim_boundary": config["claim_boundary"],
    }
    _write(summary_path, summary)
    figures = _figures(raw_rows, metrics, layer_path)
    ended = datetime.now(timezone.utc)
    outputs = (
        raw_path,
        processed_path,
        statistical_path,
        summary_path,
        decision_path,
        layer_path,
        *figures,
    )
    run_dir = ExperimentRegistry(str(ROOT / "artifacts/runs")).create(
        config,
        summary,
        simulator_version=SumoAdapter.simulator_version(),
        input_paths=(
            str(CONFIG.relative_to(ROOT)),
            config["source_osm"],
            config["source_manifest"],
            "configs/v3/frozen_thresholds.yaml",
            str(selected_path.relative_to(ROOT)),
        ),
        external_output_paths=tuple(str(path.relative_to(ROOT)) for path in outputs),
        started_at=started,
        ended_at=ended,
        source_commit=source_commit,
        source_dirty=source_dirty,
    )
    shutil.copyfile(run_dir / "manifest.json", OUTPUT / "manifest.json")
    _write(
        OUTPUT / "v3_registry.json",
        {
            "git_commit": source_commit,
            "git_dirty": source_dirty,
            "model_hash": _sha(selected_path),
            "threshold_config_hash": _sha(frozen_path),
            "result_hash": _sha(summary_path),
        },
    )
    print(summary_path)
    return summary_path


if __name__ == "__main__":
    run()
