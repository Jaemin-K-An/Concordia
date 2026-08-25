#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
from collections import defaultdict
from pathlib import Path

import networkx as nx
import sumolib
import yaml

from concordia.evaluation import summarize_selective_policy
from concordia.selective import V5DecisionInputs
from run_real_topology_study import (
    _build_network,
    _paths_are_legal,
    _qgis_layer,
    _run_one,
    _write_config,
)
from run_v3_real_topology import _actual_alignment_features, _passenger_graph, _route_overlap
from v5_frozen import load_deployment, microscopic_policy, prepare_cases, sha256, verify_frozen, write_json


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/studies/v5_real_topology"


def _select_six_od(network_path: Path) -> tuple[object, list[dict]]:
    network = sumolib.net.readNet(str(network_path))
    graph, edge_for_pair = _passenger_graph(network)
    nodes = sorted(graph)
    sampled = nodes[:: max(1, len(nodes) // 24)]
    candidates = []
    for origin in sampled:
        for destination in reversed(sampled):
            if origin == destination:
                continue
            try:
                paths = []
                for node_path in nx.shortest_simple_paths(
                    graph, origin, destination, weight="weight"
                ):
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
            if len(candidates) >= 140:
                break
        if len(candidates) >= 140:
            break
    if len(candidates) < 6:
        raise RuntimeError("real OSM topology yielded fewer than six legal OD candidates")
    selected = []
    used = set()
    for label, target in (("low", 0.15), ("medium", 0.50), ("high", 0.80)):
        for rank in range(2):
            available = [
                row
                for row in candidates
                if (row["origin"], row["destination"]) not in used
            ]
            chosen = min(
                available,
                key=lambda row: (
                    abs(row["mean_overlap"] - target),
                    row["origin"],
                    row["destination"],
                ),
            )
            selected.append({**chosen, "overlap_class": label, "stratum_rank": rank + 1})
            used.add((chosen["origin"], chosen["destination"]))
    return network, selected


def run() -> Path:
    existing = OUTPUT / "summary.json"
    if existing.is_file():
        verify_frozen()
        print(existing)
        return existing
    verify_frozen()
    config = yaml.safe_load((ROOT / "configs/v5/real_topology.yaml").read_text())
    source = ROOT / config["source_osm"]
    source_manifest = json.loads((ROOT / config["source_manifest"]).read_text())
    if sha256(source) != source_manifest["checksum_sha256"]:
        raise RuntimeError("OSM source checksum mismatch")
    regime, shift, shift_names, bundle, thresholds = load_deployment()
    policy = microscopic_policy(thresholds)
    with tempfile.TemporaryDirectory(prefix="concordia-v5-real-") as temporary:
        directory = Path(temporary)
        network_path = _build_network(directory, config)
        network, od_specs = _select_six_od(network_path)
        decisions = {}
        actual = []
        for od_index, od in enumerate(od_specs):
            od_directory = directory / f"od-{od_index}"
            od_directory.mkdir()
            sumo_config = _write_config(od_directory, network_path, od["paths"])
            for demand in config["demand_vehicles_per_hour"]:
                for seed in config["seeds"]:
                    for penetration in config["navigation_penetration"]:
                        base_features = _actual_alignment_features(
                            network,
                            od["paths"],
                            int(seed),
                            int(demand),
                            {
                                "minimum_relative_ttt_gain": 0.01,
                                "safety_delta": 0.25,
                            },
                        )
                        base_features["navigation_penetration"] = float(penetration)
                        fake_case = {
                            "case_id": f"real-od{od_index}-s{seed}-d{demand}-p{penetration:.2f}",
                            "scenario": "ring",
                            "seed": int(seed),
                            "condition": {
                                "demand_scale": int(demand) / 600.0,
                                "heterogeneity": "high",
                                "navigation_penetration": float(penetration),
                            },
                            "features": base_features,
                        }
                        prepared, prediction = prepare_cases(
                            [fake_case], regime, shift, shift_names, bundle
                        )
                        value = prepared[0]
                        decision = policy.decide(
                            V5DecisionInputs(
                                fake_case["case_id"],
                                value["regime"],
                                value["shift_class"],
                                value["domain_shift_score"],
                                float(prediction.success_probability[0]),
                                float(prediction.analytical_benefit[0]),
                                float(prediction.corrected_microscopic_benefit[0]),
                                float(prediction.microscopic_success_probability[0]),
                                float(prediction.microscopic_safety_probability_upper[0]),
                                _paths_are_legal(network, od["paths"]),
                            )
                        )
                        key = (od_index, int(seed), int(demand), float(penetration))
                        decisions[key] = {
                            "case_id": fake_case["case_id"],
                            "intervene": decision.intervene,
                            "reasons": list(decision.reasons),
                            "explanation": list(decision.explanation),
                            "regime": value["regime"],
                            "shift_class": value["shift_class"],
                            "domain_shift_score": value["domain_shift_score"],
                        }
                        run_parameters = {
                            **config,
                            "demand_vehicles_per_hour": int(demand),
                            "navigation_penetration": float(penetration),
                            "demand_provenance": "synthetic OD demand on committed real OSM geometry",
                            "tests": {"unseen": {"preference_epsilon": 0.08}},
                        }
                        for name in ("B1", "B6"):
                            row = _run_one(
                                sumo_config,
                                network,
                                od["paths"],
                                run_parameters,
                                "unseen",
                                name,
                                int(seed),
                            )
                            row.update(
                                {
                                    "od_index": od_index,
                                    "overlap_class": od["overlap_class"],
                                    "origin": od["origin"],
                                    "destination": od["destination"],
                                    "route_overlap_mean": od["mean_overlap"],
                                    "demand_vehicles_per_hour": int(demand),
                                    "navigation_penetration": float(penetration),
                                }
                            )
                            actual.append(row)
        network_hash = sha256(network_path)
        pairs = defaultdict(dict)
        for row in actual:
            key = (
                row["od_index"],
                row["seed"],
                row["demand_vehicles_per_hour"],
                row["navigation_penetration"],
            )
            pairs[key][row["policy"]] = row
        policy_rows = []
        for key, pair in sorted(pairs.items()):
            b1, b6 = pair["B1"], pair["B6"]
            gain = (b1["total_travel_time_seconds"] - b6["total_travel_time_seconds"]) / max(
                b1["total_travel_time_seconds"], 1e-9
            )
            safety_difference = b6["safety"]["cvar_drac_95"] - b1["safety"]["cvar_drac_95"]
            success = gain >= 0.01 and b6["max_regret"] <= 0.08 and safety_difference <= 0.25
            decision = decisions[key]
            policy_rows.append(
                {
                    "case_id": decision["case_id"],
                    "intervene": decision["intervene"],
                    "success": bool(decision["intervene"] and success),
                    "counterfactual_success": success,
                    "system_ttt_gain": b1["total_travel_time_seconds"]
                    - b6["total_travel_time_seconds"]
                    if decision["intervene"]
                    else 0.0,
                    "regret_violation": b6["max_regret"] > 0.08,
                    "safety_violation": safety_difference > 0.25,
                    "legal_violation": False,
                    "overlap_class": b1["overlap_class"],
                    "navigation_penetration": b1["navigation_penetration"],
                    "realized_relative_ttt_gain": gain,
                    "realized_safety_difference": safety_difference,
                    **decision,
                }
            )
        layer_path = OUTPUT / "gangnam_multi_od_v5_delta.geojson"
        qgis_rows = [{**row, "mode": "transfer"} for row in actual]
        _qgis_layer(network, qgis_rows, layer_path)
    metrics = summarize_selective_policy(policy_rows)
    by_overlap = {
        label: summarize_selective_policy(
            [row for row in policy_rows if row["overlap_class"] == label]
        )
        for label in config["overlap_strata"]
    }
    summary = {
        "complete": True,
        "study": "v5 frozen stratified multi-OD real topology",
        "od_pair_count": len(od_specs),
        "paired_condition_count": len(policy_rows),
        "source_osm_sha256": source_manifest["checksum_sha256"],
        "sumo_network_sha256": network_hash,
        "all_routes_legal": all(
            _paths_are_legal(network, od["paths"]) for od in od_specs
        ),
        "od_pairs": [
            {key: value for key, value in od.items() if key != "paths"}
            for od in od_specs
        ],
        "primary_metrics": metrics,
        "activation_by_overlap_class": by_overlap,
        "real_geometry_synthetic_demand": True,
        "claim_boundary": config["claim_boundary"],
    }
    write_json(OUTPUT / "raw_metrics.json", actual)
    write_json(OUTPUT / "policy_rows.json", policy_rows)
    write_json(OUTPUT / "decision_log.json", list(decisions.values()))
    write_json(OUTPUT / "summary.json", summary)
    print(OUTPUT / "summary.json")
    return OUTPUT / "summary.json"


if __name__ == "__main__":
    run()
