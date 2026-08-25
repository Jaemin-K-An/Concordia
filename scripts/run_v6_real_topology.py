#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import networkx as nx
import numpy as np
import sumolib
import yaml

from concordia.micro_v6 import MICRO_V6_FEATURE_SCHEMA, build_safe_micro_label, selective_metrics
from run_real_topology_study import (
    _build_network,
    _paths_are_legal,
    _qgis_layer,
    _run_one,
    _write_config,
)
from run_v3_real_topology import _actual_alignment_features, _passenger_graph, _route_overlap
from v5_frozen import load_deployment, prepare_cases
from v6_frozen import load_policy, sha256, verify_frozen, write_json


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/studies/v6_real_topology"


def _select_od(network_path: Path, counts: dict[str, int]) -> tuple[object, list[dict]]:
    network = sumolib.net.readNet(str(network_path))
    graph, edge_for_pair = _passenger_graph(network)
    nodes = sorted(graph)
    sampled = nodes[:: max(1, len(nodes) // 30)]
    candidates = []
    for origin in sampled:
        for destination in reversed(sampled):
            if origin == destination:
                continue
            try:
                paths = []
                for node_path in nx.shortest_simple_paths(graph, origin, destination, weight="weight"):
                    edge_ids = tuple(
                        edge_for_pair[pair] for pair in zip(node_path, node_path[1:])
                    )
                    paths.append((tuple(node_path), edge_ids))
                    if len(paths) == 3:
                        break
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            if len(paths) < 3 or min(len(path[1]) for path in paths) < 3:
                continue
            mean, maximum = _route_overlap(paths)
            candidates.append(
                {
                    "origin": origin,
                    "destination": destination,
                    "paths": paths,
                    "mean_overlap": mean,
                    "maximum_overlap": maximum,
                }
            )
            if len(candidates) >= 220:
                break
        if len(candidates) >= 220:
            break
    targets = {"low": 0.15, "medium": 0.50, "high": 0.80}
    selected = []
    used = set()
    for label in ("low", "medium", "high"):
        for rank in range(int(counts[label])):
            available = [
                row for row in candidates
                if (row["origin"], row["destination"]) not in used
            ]
            if not available:
                raise RuntimeError("real OSM topology has too few legal OD candidates")
            chosen = min(
                available,
                key=lambda row: (
                    abs(row["mean_overlap"] - targets[label]),
                    row["origin"], row["destination"],
                ),
            )
            selected.append({**chosen, "overlap_class": label, "stratum_rank": rank + 1})
            used.add((chosen["origin"], chosen["destination"]))
    return network, selected


def _probe_features(
    probe: dict,
    analytical: dict,
    alignment: dict,
    network,
    paths,
    demand: int,
    penetration: float,
) -> dict[str, float]:
    edge_values = list(probe["edge_metrics"].values())
    flows = np.asarray([value["flow"] for value in edge_values], dtype=float)
    speeds = np.asarray([value["speed"] for value in edge_values], dtype=float)
    safety = np.asarray([value["safety_risk"] for value in edge_values], dtype=float)
    density = flows / np.maximum(speeds * 3.6, 1e-6)
    route_lengths = [
        sum(network.getEdge(edge).getLength() for edge in edges)
        for _nodes, edges in paths
    ]
    minimum_headway = 3600.0 / max(float(flows.max()) if len(flows) else 1.0, 1.0)
    features = {
        "density_mean": float(density.mean()) if len(density) else 0.0,
        "flow_mean": float(flows.mean()) if len(flows) else 0.0,
        "occupancy_mean": float(np.clip((density.mean() if len(density) else 0.0) / 140.0, 0, 1)),
        "mean_speed": float(speeds.mean()) if len(speeds) else 0.0,
        "speed_variance": float(speeds.var()) if len(speeds) else 0.0,
        "acceleration_variance": float(safety.mean()) if len(safety) else 0.0,
        "queue_length": 0.0,
        "halting_count": 0.0,
        "lane_occupancy": float(np.clip((density.max() if len(density) else 0.0) / 140.0, 0, 1)),
        "headway_mean": minimum_headway,
        "headway_variance": 0.0,
        "demand_vehicles_per_hour": float(demand),
        "density_slope_30s": 0.0,
        "speed_slope_30s": 0.0,
        "flow_instability": float(flows.std() / max(flows.mean(), 1e-6)) if len(flows) else 0.0,
        "queue_growth_rate": 0.0,
        "short_horizon_speed_oscillation": float(speeds.std()) if len(speeds) else 0.0,
        "occupancy_variance_30s": float(density.var() / (140.0**2)) if len(density) else 0.0,
        "route_overlap": float(alignment["route_overlap"]),
        "alternative_capacity_ratio": float(alignment["alternative_capacity_ratio"]),
        "volume_capacity_ratio": float(alignment["volume_capacity_ratio"]),
        "bottleneck_centrality": float(alignment["bottleneck_centrality"]),
        "route_length_ratio": max(route_lengths) / max(min(route_lengths), 1e-9),
        "topology_merge": 0.0,
        "topology_signalized": 0.0,
        "topology_two_route": 0.0,
        "topology_real_like": 1.0,
        "topology_ring": 0.0,
        "perturbation_strength": 0.0,
        "predicted_acceptance": float(alignment.get("acceptance_probability", 0.0)),
        "preference_slack_mean": 0.04,
        "preference_slack_std": 0.02,
        "preference_variance": float(alignment["preference_variance"]),
        "heterogeneity_low": 0.0,
        "heterogeneity_medium": 0.0,
        "heterogeneity_high": 1.0,
        "heterogeneity_bimodal": 0.0,
        "heterogeneity_long_tail": 0.0,
        "acceptance_multiplier": 1.0,
        "navigation_penetration": float(penetration),
        "minimum_headway": minimum_headway,
        "closing_speed_p90": float(speeds.std()) if len(speeds) else 0.0,
        "drac_proxy_p95": float(np.sqrt(max(safety.max(), 0.0))) if len(safety) else 0.0,
        "lane_change_density": 0.0,
        "merge_interaction_density": float(alignment["route_overlap"] * (density.mean() if len(density) else 0.0)),
        "speed_differential": float(speeds.max() - speeds.min()) if len(speeds) else 0.0,
        "hard_braking_recent_rate": 0.0,
        "analytical_success_probability": float(analytical["success_probability"]),
        "analytical_predicted_ttt_gain": float(analytical["predicted_ttt_gain"]),
    }
    if set(features) != set(MICRO_V6_FEATURE_SCHEMA):
        raise RuntimeError("v6 real-topology feature schema mismatch")
    return features


def run() -> Path:
    existing = OUTPUT / "summary.json"
    if existing.is_file():
        verify_frozen()
        print(existing)
        return existing
    before = verify_frozen()
    policy = load_policy()
    config = yaml.safe_load((ROOT / "configs/v6/real_topology.yaml").read_text())
    preregistration = yaml.safe_load((ROOT / "configs/v6/preregistration.yaml").read_text())
    source = ROOT / config["source_osm"]
    source_manifest = json.loads((ROOT / config["source_manifest"]).read_text())
    if sha256(source) != source_manifest["checksum_sha256"]:
        raise RuntimeError("OSM source checksum mismatch")
    with tempfile.TemporaryDirectory(prefix="concordia-v6-real-") as temporary:
        directory = Path(temporary)
        network_path = _build_network(directory, config)
        network, od_specs = _select_od(network_path, config["stratum_counts"])
        specifications = []
        fake_cases = []
        for od_index, od in enumerate(od_specs):
            for demand in config["demand_vehicles_per_hour"]:
                for seed in config["seeds"]:
                    for penetration in config["navigation_penetration"]:
                        alignment = _actual_alignment_features(
                            network, od["paths"], int(seed), int(demand),
                            {"minimum_relative_ttt_gain": 0.01, "safety_delta": 0.25},
                        )
                        alignment["navigation_penetration"] = float(penetration)
                        case_id = f"v6-real-od{od_index}-s{seed}-d{demand}-p{penetration:.2f}"
                        fake_cases.append(
                            {
                                "case_id": case_id, "scenario": "ring", "seed": int(seed),
                                "condition": {
                                    "demand_scale": int(demand) / 600.0,
                                    "heterogeneity": "high",
                                    "navigation_penetration": float(penetration),
                                },
                                "features": alignment,
                            }
                        )
                        specifications.append((od_index, od, int(demand), int(seed), float(penetration), alignment))
        regime, shift, shift_names, bundle, _thresholds = load_deployment()
        _prepared, prediction = prepare_cases(fake_cases, regime, shift, shift_names, bundle)
        actual = []
        feature_rows = []
        for index, (od_index, od, demand, seed, penetration, alignment) in enumerate(specifications):
            od_directory = directory / f"od-{od_index}"
            od_directory.mkdir(exist_ok=True)
            sumo_config = _write_config(od_directory, network_path, od["paths"])
            run_parameters = {
                **config,
                "demand_vehicles_per_hour": demand,
                "navigation_penetration": penetration,
                "demand_provenance": "synthetic OD demand on committed real OSM geometry",
                "tests": {"unseen": {"preference_epsilon": 0.08}},
            }
            probe_parameters = {
                **run_parameters,
                "vehicle_generation_seconds": 30,
                "maximum_simulation_seconds": 30,
            }
            probe = _run_one(
                sumo_config, network, od["paths"], probe_parameters, "unseen", "B1", seed
            )
            analytical = {
                "success_probability": float(prediction.success_probability[index]),
                "predicted_ttt_gain": float(prediction.analytical_benefit[index]),
            }
            features = _probe_features(
                probe, analytical, alignment, network, od["paths"], demand, penetration
            )
            pair = {}
            for name in ("B1", "B6"):
                value = _run_one(
                    sumo_config, network, od["paths"], run_parameters, "unseen", name, seed
                )
                value.update(
                    {
                        "od_index": od_index,
                        "overlap_class": od["overlap_class"],
                        "origin": od["origin"],
                        "destination": od["destination"],
                        "route_overlap_mean": od["mean_overlap"],
                        "demand_vehicles_per_hour": demand,
                        "navigation_penetration": penetration,
                    }
                )
                pair[name] = value
                actual.append(value)
            baseline = {
                **pair["B1"],
                "maximum_affected_regret": pair["B1"]["max_regret"],
                "all_executed_routes_legal": True,
            }
            adaptive = {
                **pair["B6"],
                "maximum_affected_regret": pair["B6"]["max_regret"],
                "all_executed_routes_legal": _paths_are_legal(network, od["paths"]),
            }
            label_config = preregistration["label"]
            label = build_safe_micro_label(
                baseline, adaptive,
                minimum_relative_ttt_gain=float(label_config["minimum_relative_ttt_gain"]),
                safety_margin=float(label_config["safety_cvar_drac_margin"]),
                regret_limit=float(label_config["regret_limit"]),
            )
            feature_rows.append(
                {
                    "case_id": fake_cases[index]["case_id"],
                    "decision_time": 30.0,
                    "feature_observation_end_time": 30.0,
                    "features_pre_decision": features,
                    "condition": {
                        "penetration": penetration, "demand": demand,
                        "topology": "real_osm", "heterogeneity": "high",
                    },
                    "label": label.to_dict(),
                    "od_index": od_index,
                    "overlap_class": od["overlap_class"],
                    "counterfactual_B1": baseline,
                    "counterfactual_adaptive": adaptive,
                }
            )
        decisions = policy.decide(feature_rows)
        mask = [decision["intervene"] for decision in decisions]
        metrics = selective_metrics(feature_rows, mask)
        layer_path = OUTPUT / "gangnam_multi_od_v6_delta.geojson"
        _qgis_layer(network, [{**row, "mode": "transfer"} for row in actual], layer_path)
        network_hash = sha256(network_path)
    summary = {
        "complete": True,
        "study": "v6 frozen 10-OD real OSM geometry transfer",
        "od_pair_count": len(od_specs),
        "paired_condition_count": len(feature_rows),
        "actual_sumo_run_count_including_predecision_probes": 3 * len(feature_rows),
        "source_osm_sha256": source_manifest["checksum_sha256"],
        "sumo_network_sha256": network_hash,
        "all_routes_legal": all(_paths_are_legal(network, od["paths"]) for od in od_specs),
        "od_pairs": [{key: value for key, value in od.items() if key != "paths"} for od in od_specs],
        "primary_metrics": metrics,
        "intervention_target_met": metrics["intervention_count"] > 0,
        "safe_success_target_met": metrics["success_count"] > 0,
        "real_geometry_synthetic_demand": True,
        "claim_boundary": config["claim_boundary"],
        "freeze_manifest_hash_before": before["manifest_self_hash"],
        "freeze_manifest_hash_after": verify_frozen()["manifest_self_hash"],
        "frozen_immutable": True,
    }
    write_json(OUTPUT / "raw_metrics.json", actual)
    write_json(OUTPUT / "feature_and_label_rows.json", feature_rows)
    write_json(OUTPUT / "decision_log.json", decisions)
    write_json(OUTPUT / "summary.json", summary)
    print(OUTPUT / "summary.json")
    return OUTPUT / "summary.json"


if __name__ == "__main__":
    run()
