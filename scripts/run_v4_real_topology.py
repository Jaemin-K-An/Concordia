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

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/concordia-matplotlib-v4")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/concordia-cache-v4")

import matplotlib
import networkx as nx
import numpy as np
import sumolib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from concordia.evaluation import ExperimentRegistry, capture_source_state, summarize_selective_policy
from concordia.feasibility import V4PredictionBundle, V4_FEATURE_SCHEMA, expand_v4_features
from concordia.selective import PrecisionConstrainedPolicy, V4DecisionInputs
from concordia.simulation import SumoAdapter
from run_real_topology_study import _build_network, _paths_are_legal, _qgis_layer, _run_one, _write_config
from run_v3_real_topology import _actual_alignment_features, _passenger_graph, _route_overlap


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/v4/real_topology.yaml"
VALIDATION = ROOT / "artifacts/studies/v4_precision_validation"
OUTPUT = ROOT / "artifacts/studies/v4_real_topology"


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _select_od_classes(network_path: Path, development_od: list[str]) -> tuple[object, list[dict]]:
    network = sumolib.net.readNet(str(network_path))
    graph, edge_for_pair = _passenger_graph(network)
    nodes = sorted(graph)
    stride = max(1, len(nodes) // 22)
    sampled = nodes[::stride]
    candidates = []
    for origin in sampled:
        for destination in reversed(sampled):
            if origin == destination or [origin, destination] == development_od:
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
            overlap, maximum = _route_overlap(paths)
            candidates.append(
                {
                    "origin": origin,
                    "destination": destination,
                    "paths": paths,
                    "mean_overlap": overlap,
                    "maximum_overlap": maximum,
                }
            )
            if len(candidates) >= 100:
                break
        if len(candidates) >= 100:
            break
    if len(candidates) < 3:
        raise RuntimeError("real topology did not yield three legal alternative OD candidates")
    selected = []
    used = set()
    for label, target in (("low", 0.15), ("medium", 0.50), ("high", 0.80)):
        available = [
            row
            for row in candidates
            if (row["origin"], row["destination"]) not in used
        ]
        chosen = min(available, key=lambda row: abs(row["mean_overlap"] - target))
        chosen = {**chosen, "overlap_class": label}
        selected.append(chosen)
        used.add((chosen["origin"], chosen["destination"]))
    return network, selected


def _policy(selection: dict) -> PrecisionConstrainedPolicy:
    name = selection["selected_policy"]
    point = selection["policy_operating_points"][name]
    return PrecisionConstrainedPolicy(
        name,
        probability_threshold=float(point["score_threshold"]) if name != "V4-E" else 0.0,
        benefit_threshold=float(point["benefit_threshold"]),
        safety_delta=float(selection["safety_delta"]),
        safety_probability_threshold=float(
            selection["safety_failure_probability_threshold"]
        ),
        esiv_threshold=float(point["score_threshold"]) if name == "V4-E" else 0.0,
        use_esiv=name == "V4-E",
    )


def _features(network, paths, seed: int, demand: int, frozen: dict) -> dict:
    old = _actual_alignment_features(network, paths, seed, demand, frozen)
    fake = {
        "scenario": "ring",
        "seed": seed,
        "condition": {
            "demand_scale": demand / 600.0,
            "heterogeneity": "high",
            "navigation_penetration": 1.0,
        },
        "features": old,
    }
    features = expand_v4_features(fake)
    route_values = []
    for _nodes, edge_ids in paths:
        edges = [network.getEdge(edge_id) for edge_id in edge_ids]
        route_values.append(
            {
                "eta": sum(edge.getLength() / max(edge.getSpeed(), 1e-9) for edge in edges),
                "capacity": min(edge.getLaneNumber() * 1800.0 for edge in edges),
            }
        )
    eta = np.asarray([row["eta"] for row in route_values], dtype=float)
    features.update(
        {
            "route_count": float(len(paths)),
            "eta_dispersion": float(eta.std() / max(eta.mean(), 1e-9)),
            "bottleneck_load": features["volume_capacity_ratio"]
            * (1.0 + features["bottleneck_centrality"]),
            "aps_alternative_capacity_interaction": features["alignment_potential_score"]
            * features["alternative_capacity_ratio"],
            "route_overlap_demand_interaction": features["route_overlap"]
            * features["demand"],
            "poa_penetration_interaction": features["price_of_anarchy"],
            "safety_margin_demand_interaction": features["safety_margin"]
            * features["demand"],
        }
    )
    return {name: float(features[name]) for name in V4_FEATURE_SCHEMA}


def run() -> Path:
    source_commit, source_dirty = capture_source_state()
    started = datetime.now(timezone.utc)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    source_path = ROOT / config["source_osm"]
    source_manifest = json.loads(
        (ROOT / config["source_manifest"]).read_text(encoding="utf-8")
    )
    if _sha(source_path) != source_manifest["checksum_sha256"]:
        raise RuntimeError("OSM source does not match provenance checksum")
    frozen_model = ROOT / "configs/v4/frozen_model.yaml"
    frozen_thresholds = ROOT / "configs/v4/frozen_thresholds.yaml"
    model_hash = _sha(frozen_model)
    threshold_hash = _sha(frozen_thresholds)
    frozen = yaml.safe_load(frozen_thresholds.read_text(encoding="utf-8"))
    probability = json.loads(
        (VALIDATION / "probability_package.json").read_text(encoding="utf-8")
    )
    benefit = json.loads((VALIDATION / "benefit_package.json").read_text(encoding="utf-8"))
    safety = json.loads((VALIDATION / "safety_package.json").read_text(encoding="utf-8"))
    selection = json.loads(
        (VALIDATION / "threshold_selection.json").read_text(encoding="utf-8")
    )
    bundle = V4PredictionBundle.from_packages(probability, benefit, safety)
    policy = _policy(selection)
    with tempfile.TemporaryDirectory(prefix="concordia-v4-real-") as temporary:
        directory = Path(temporary)
        network_path = _build_network(directory, config)
        network, od_specs = _select_od_classes(network_path, config["development_od"])
        actual = []
        decisions = {}
        for od_index, od_spec in enumerate(od_specs):
            od_directory = directory / f"od-{od_index}"
            od_directory.mkdir()
            run_config = _write_config(od_directory, network_path, od_spec["paths"])
            for demand in config["demand_vehicles_per_hour"]:
                run_parameters = {
                    **config,
                    "demand_vehicles_per_hour": int(demand),
                    "demand_provenance": "synthetic multi-OD demand on real OSM geometry",
                    "tests": {"unseen": {"preference_epsilon": 0.08}},
                }
                for seed in config["seeds"]:
                    features = _features(network, od_spec["paths"], int(seed), int(demand), frozen)
                    matrix = np.asarray(
                        [[features[name] for name in V4_FEATURE_SCHEMA]], dtype=float
                    )
                    prediction = bundle.predict(matrix)
                    inputs = V4DecisionInputs(
                        case_id=f"real-{od_spec['overlap_class']}-s{seed}-d{demand}",
                        success_probability=float(prediction["probability"][0]),
                        success_probability_lower=float(prediction["probability_lower"][0]),
                        expected_benefit=float(prediction["expected_benefit"][0]),
                        benefit_lower=float(prediction["benefit_lower"][0]),
                        safety_difference_upper=float(
                            prediction["safety_difference_upper"][0]
                        ),
                        safety_failure_probability=float(
                            prediction["safety_failure_probability"][0]
                        ),
                        safety_failure_probability_upper=float(
                            prediction["safety_failure_probability_upper"][0]
                        ),
                        esiv=float(prediction["esiv"][0]),
                        esiv_lower=float(prediction["esiv_lower"][0]),
                        legal=_paths_are_legal(network, od_spec["paths"]),
                    )
                    key = (od_spec["overlap_class"], int(seed), int(demand))
                    decisions[key] = policy.decide(inputs)
                    for policy_name in ("B1", "B6"):
                        row = _run_one(
                            run_config,
                            network,
                            od_spec["paths"],
                            run_parameters,
                            "unseen",
                            policy_name,
                            int(seed),
                        )
                        row.update(
                            {
                                "overlap_class": od_spec["overlap_class"],
                                "origin": od_spec["origin"],
                                "destination": od_spec["destination"],
                                "route_overlap_mean": od_spec["mean_overlap"],
                                "route_overlap_maximum": od_spec["maximum_overlap"],
                                "demand_vehicles_per_hour": int(demand),
                            }
                        )
                        actual.append(row)
        network_hash = _sha(network_path)
    pairs = defaultdict(dict)
    for row in actual:
        key = (row["overlap_class"], row["seed"], row["demand_vehicles_per_hour"])
        pairs[key][row["policy"]] = row
    policy_rows = {"B6": [], "V4-F": []}
    selected_actual = []
    decision_log = []
    for key, pair in sorted(pairs.items()):
        decision = decisions[key]
        b1, b6 = pair["B1"], pair["B6"]
        gain = (b1["total_travel_time_seconds"] - b6["total_travel_time_seconds"]) / max(
            b1["total_travel_time_seconds"], 1e-9
        )
        safety_difference = b6["safety"]["cvar_drac_95"] - b1["safety"]["cvar_drac_95"]
        success = gain >= 0.01 and b6["max_regret"] <= 0.08 + 1e-10 and safety_difference <= 0.25
        selected_actual.append(
            {**(b6 if decision.intervene else b1), "policy": "V4-F"}
        )
        for name, intervene in (("B6", True), ("V4-F", decision.intervene)):
            selected_ttt = b6["total_travel_time_seconds"] if intervene else b1[
                "total_travel_time_seconds"
            ]
            policy_rows[name].append(
                {
                    "case_id": decision.case_id,
                    "intervene": intervene,
                    "success": bool(intervene and success),
                    "counterfactual_success": success,
                    "system_ttt_gain": b1["total_travel_time_seconds"] - selected_ttt,
                    "regret_violation": b6["max_regret"] > 0.08 + 1e-10,
                    "safety_violation": safety_difference > 0.25,
                    "legal_violation": False,
                    "overlap_class": b1["overlap_class"],
                }
            )
        decision_log.append(
            {
                **decision.__dict__,
                "outcome": "SUCCESS" if decision.intervene and success else "FAILURE" if decision.intervene else "ABSTAIN",
                "overlap_class": b1["overlap_class"],
                "realized_relative_ttt_gain": gain,
                "realized_safety_difference": safety_difference,
                "model_hash": model_hash,
                "threshold_hash": threshold_hash,
            }
        )
    metrics = {name: summarize_selective_policy(values) for name, values in policy_rows.items()}
    by_overlap = {
        overlap: summarize_selective_policy(
            [row for row in policy_rows["V4-F"] if row["overlap_class"] == overlap]
        )
        for overlap in config["od_overlap_classes"]
    }
    summary = {
        "complete": True,
        "study": "Study XIV — Multi-OD Real Topology",
        "simulator_version": SumoAdapter.simulator_version(),
        "source_osm_sha256": source_manifest["checksum_sha256"],
        "sumo_network_sha256": network_hash,
        "od_pairs": [
            {
                key: value
                for key, value in od.items()
                if key not in {"paths"}
            }
            for od in od_specs
        ],
        "all_routes_legal": all(_paths_are_legal(network, od["paths"]) for od in od_specs),
        "policy_metrics": metrics,
        "activation_by_overlap_class": by_overlap,
        "claim_boundary": config["claim_boundary"],
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    raw_path = OUTPUT / "raw_metrics.json"
    processed_path = OUTPUT / "processed_metrics.json"
    summary_path = OUTPUT / "summary.json"
    decision_path = OUTPUT / "decision_log.json"
    layer_path = OUTPUT / "gangnam_multi_od_v4_delta.geojson"
    _write(raw_path, actual + selected_actual)
    _write(processed_path, {"metrics": metrics, "by_overlap": by_overlap})
    _write(summary_path, summary)
    _write(decision_path, decision_log)
    qgis_rows = []
    for row in actual:
        if row["policy"] == "B1":
            qgis_rows.append({**row, "mode": "transfer"})
    for row in selected_actual:
        qgis_rows.append({**row, "mode": "transfer", "policy": "B6"})
    _qgis_layer(network, qgis_rows, layer_path)
    layer = json.loads(layer_path.read_text(encoding="utf-8"))
    layer["name"] = "gangnam_multi_od_v4_policy_delta"
    layer["od_pairs"] = [
        [od["origin"], od["destination"], od["overlap_class"]] for od in od_specs
    ]
    _write(layer_path, layer)
    figure_dir = OUTPUT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(6.5, 4.2))
    axis.bar(
        list(by_overlap),
        [by_overlap[name]["coverage"] for name in by_overlap],
        color=["#79aa98", "#327a65", "#164c3d"],
    )
    axis.set(ylim=(0, 1), ylabel="V4-F activation coverage", xlabel="Route overlap class")
    fig.tight_layout()
    figure_path = figure_dir / "activation_by_overlap.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    ended = datetime.now(timezone.utc)
    outputs = (raw_path, processed_path, summary_path, decision_path, layer_path, figure_path)
    registry = ExperimentRegistry(str(ROOT / "artifacts/runs")).create(
        config,
        summary,
        simulator_version=SumoAdapter.simulator_version(),
        input_paths=(
            "configs/v4/real_topology.yaml",
            config["source_osm"],
            config["source_manifest"],
            "configs/v4/frozen_model.yaml",
            "configs/v4/frozen_thresholds.yaml",
        ),
        external_output_paths=tuple(str(path.relative_to(ROOT)) for path in outputs),
        started_at=started,
        ended_at=ended,
        source_commit=source_commit,
        source_dirty=source_dirty,
    )
    shutil.copyfile(registry / "manifest.json", OUTPUT / "manifest.json")
    _write(
        OUTPUT / "v4_registry.json",
        {
            "git_commit": source_commit,
            "git_dirty": source_dirty,
            "model_hash": model_hash,
            "threshold_hash": threshold_hash,
            "result_hash": _sha(summary_path),
        },
    )
    print(summary_path)
    return summary_path


if __name__ == "__main__":
    run()
