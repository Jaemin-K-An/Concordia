#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from concordia.safety_v8.features import action_aware_features
from concordia.selective_v8.metrics import filtered_policy_metrics, ranking_metrics
from concordia.uplift_v7.evaluation import deployment_metrics
from concordia.uplift_v7.outcomes import paired_treatment_outcomes
from concordia.uplift_v7.paired_dataset import enrich_predecision_features
from run_real_topology_study import _build_network, _paths_are_legal, _qgis_layer, _run_one, _write_config
from run_v3_real_topology import _actual_alignment_features
from run_v6_real_topology import _probe_features, _select_od
from v5_frozen import load_deployment, prepare_cases
from v6_frozen import load_policy as load_v6_policy
from v7_frozen import load_policy as load_v7_policy
from v8_common import ROOT, sha256, write_json
from v8_frozen import load_policy, verify_frozen


OUTPUT = ROOT / "artifacts/studies/v8_real_topology"


def _new_od_specs(network_path: Path, counts: dict[str, int]):
    prior = json.loads((ROOT / "artifacts/studies/v7_real_topology/summary.json").read_text())
    excluded = {(row["origin"], row["destination"]) for row in prior["od_pairs"]}
    expanded = {key: int(value) + 5 for key, value in counts.items()}
    network, candidates = _select_od(network_path, expanded)
    selected = []
    per_stratum = Counter()
    for candidate in candidates:
        key = (candidate["origin"], candidate["destination"])
        stratum = candidate["overlap_class"]
        if key in excluded or per_stratum[stratum] >= int(counts[stratum]):
            continue
        selected.append(candidate)
        per_stratum[stratum] += 1
    if per_stratum != Counter({key: int(value) for key, value in counts.items()}):
        raise RuntimeError(f"not enough new OSM OD pairs after v7 exclusion: {per_stratum}")
    return network, selected, excluded


def run():
    summary_path = OUTPUT / "summary.json"
    if summary_path.is_file():
        verify_frozen()
        print(summary_path)
        return summary_path
    before = verify_frozen()
    policy = load_policy()
    v7 = load_v7_policy()
    config = yaml.safe_load((ROOT / "configs/v8/real_topology.yaml").read_text())
    source = ROOT / config["source_osm"]
    source_manifest = json.loads((ROOT / config["source_manifest"]).read_text())
    if sha256(source) != source_manifest["checksum_sha256"]:
        raise RuntimeError("OSM source checksum mismatch")
    with tempfile.TemporaryDirectory(prefix="concordia-v8-real-") as temporary:
        directory = Path(temporary)
        network_path = _build_network(directory, config)
        network, od_specs, excluded = _new_od_specs(network_path, config["stratum_counts"])
        specifications = []
        analytical_cases = []
        for od_index, od in enumerate(od_specs):
            for demand in config["demand_vehicles_per_hour"]:
                for seed in config["seeds"]:
                    for penetration in config["navigation_penetration"]:
                        alignment = _actual_alignment_features(
                            network, od["paths"], int(seed), int(demand),
                            {"minimum_relative_ttt_gain": 0.01, "safety_delta": 0.25},
                        )
                        alignment["navigation_penetration"] = float(penetration)
                        case_id = f"v8-real-od{od_index}-s{seed}-d{demand}-p{penetration:.2f}"
                        analytical_cases.append({
                            "case_id": case_id,
                            "scenario": "ring",
                            "seed": int(seed),
                            "condition": {
                                "demand_scale": int(demand) / 600.0,
                                "heterogeneity": "high",
                                "navigation_penetration": float(penetration),
                            },
                            "features": alignment,
                        })
                        specifications.append((od_index, od, int(demand), int(seed), float(penetration), alignment))
        if len(specifications) != 120:
            raise RuntimeError("v8 OSM transfer requires exactly 120 paired conditions")
        regime, shift, shift_names, bundle, _ = load_deployment()
        _prepared, analytical_prediction = prepare_cases(
            analytical_cases, regime, shift, shift_names, bundle
        )
        actual_runs, feature_rows, pairs = [], [], []
        for index, (od_index, od, demand, seed, penetration, alignment) in enumerate(specifications):
            od_directory = directory / f"od-{od_index}"
            od_directory.mkdir(exist_ok=True)
            sumo_config = _write_config(od_directory, network_path, od["paths"])
            parameters = {
                **config,
                "demand_vehicles_per_hour": demand,
                "navigation_penetration": penetration,
                "demand_provenance": "synthetic OD demand on committed real OSM geometry",
                "tests": {"unseen": {"preference_epsilon": 0.08}},
            }
            probe = _run_one(
                sumo_config, network, od["paths"],
                {**parameters, "vehicle_generation_seconds": 30, "maximum_simulation_seconds": 30},
                "unseen", "B1", seed,
            )
            analytical = {
                "success_probability": float(analytical_prediction.success_probability[index]),
                "predicted_ttt_gain": float(analytical_prediction.analytical_benefit[index]),
            }
            features = _probe_features(
                probe, analytical, alignment, network, od["paths"], demand, penetration
            )
            pair = {}
            for name in ("B1", "B6"):
                value = _run_one(sumo_config, network, od["paths"], parameters, "unseen", name, seed)
                value.update({
                    "od_index": od_index,
                    "overlap_class": od["overlap_class"],
                    "origin": od["origin"],
                    "destination": od["destination"],
                    "route_overlap_mean": od["mean_overlap"],
                    "demand_vehicles_per_hour": demand,
                    "navigation_penetration": penetration,
                })
                pair[name] = value
                actual_runs.append(value)
            baseline = {**pair["B1"], "maximum_affected_regret": pair["B1"]["max_regret"], "all_executed_routes_legal": True}
            adaptive = {
                **pair["B6"],
                "maximum_affected_regret": pair["B6"]["max_regret"],
                "all_executed_routes_legal": _paths_are_legal(network, od["paths"]),
            }
            condition = {
                "penetration": penetration,
                "demand": demand,
                "topology": "real_like",
                "heterogeneity": "high",
                "acceptance_multiplier": 1.0,
                "perturbation": "none",
            }
            case_id = analytical_cases[index]["case_id"]
            feature_rows.append({"case_id": case_id, "features_pre_decision": features})
            pairs.append((case_id, condition, od_index, od, baseline, adaptive, features))
        v6_scores, _, _ = load_v6_policy().probabilities(feature_rows)
        rows = []
        for score, pair in zip(v6_scores, pairs):
            case_id, condition, od_index, od, baseline, adaptive, features = pair
            outcomes = paired_treatment_outcomes(baseline, adaptive)
            rows.append({
                "pair_id": case_id,
                "source": "v8_new_real_osm_geometry_synthetic_demand",
                "seed": int(baseline["seed"]),
                "condition": condition,
                "od_index": od_index,
                "overlap_class": od["overlap_class"],
                "decision_time": 30.0,
                "feature_observation_end_time": 30.0,
                "predecision_features": enrich_predecision_features(
                    features, condition, v6_micro_success_score=float(score), number_route_alternatives=3
                ),
                "ttt_b1": float(baseline["total_travel_time_seconds"]),
                "ttt_adaptive": float(adaptive["total_travel_time_seconds"]),
                "risk_b1": float(baseline["safety"]["cvar_drac_95"]),
                "risk_adaptive": float(adaptive["safety"]["cvar_drac_95"]),
                "generated_vehicle_count": int(baseline["generated_vehicle_count"]),
                "outcomes": outcomes.to_dict(),
                "counterfactual_B1": baseline,
                "counterfactual_adaptive": adaptive,
                "legal_executable_predecision": True,
                "pairing": {
                    "same_seed": baseline["seed"] == adaptive["seed"],
                    "same_osm_network": True,
                    "common_od_demand": True,
                    "metadata_identical_except_treatment": True,
                    "treatment_only_difference": True,
                },
            })
        traffic = policy.traffic_ranker.predict(rows)
        percentiles = policy.traffic_ranker.percentiles(traffic)
        for row, score, percentile in zip(rows, traffic, percentiles):
            row["v8_features"] = action_aware_features(
                row, traffic_uplift_score=float(score), traffic_rank_percentile=float(percentile)
            )
            row["unsafe_intervention"] = int(float(row["outcomes"]["tau_s"]) > 0.25)
        decisions = policy.decide(rows)
        mask = np.asarray([decision["intervene"] for decision in decisions])
        screen = percentiles >= policy.traffic_rank_percentile_cutoff
        metrics = filtered_policy_metrics(rows, mask, screen)
        v7_mask = [decision["intervene"] for decision in v7.decide(rows)]
        comparison = {
            "B1": deployment_metrics(rows, [False] * len(rows)),
            "B6": deployment_metrics(rows, [True] * len(rows)),
            "V7-F": deployment_metrics(rows, v7_mask),
            "V8-traffic-only": deployment_metrics(rows, screen),
            "V8-F": metrics,
        }
        layer_path = OUTPUT / "gangnam_multi_od_v8_delta.geojson"
        _qgis_layer(network, [{**row, "mode": "transfer"} for row in actual_runs], layer_path)
        network_hash = sha256(network_path)
    after = verify_frozen()
    summary = {
        "complete": True,
        "study": "v8 frozen 15-new-OD real OSM geometry safety-filtered transfer",
        "od_pair_count": len(od_specs),
        "prior_v7_od_pair_overlap_count": sum((od["origin"], od["destination"]) in excluded for od in od_specs),
        "overlap_stratum_counts": Counter(od["overlap_class"] for od in od_specs),
        "paired_condition_count": len(rows),
        "actual_sumo_run_count_including_predecision_probes": 3 * len(rows),
        "source_osm_sha256": source_manifest["checksum_sha256"],
        "sumo_network_sha256": network_hash,
        "all_routes_legal": all(_paths_are_legal(network, od["paths"]) for od in od_specs),
        "od_pairs": [{key: value for key, value in od.items() if key != "paths"} for od in od_specs],
        "primary_metrics": metrics,
        "architecture_comparison": comparison,
        "traffic_ranking": ranking_metrics(rows, traffic),
        "intervention_target_met": metrics["intervention_count"] > 0,
        "safe_success_target_met": metrics["success_count"] > 0,
        "real_geometry_synthetic_demand": True,
        "claim_boundary": config["claim_boundary"],
        "freeze_manifest_hash_before": before["manifest_self_hash"],
        "freeze_manifest_hash_after": after["manifest_self_hash"],
        "frozen_immutable": before["manifest_self_hash"] == after["manifest_self_hash"],
        "rl_used": False,
    }
    write_json(OUTPUT / "raw_metrics.json", actual_runs)
    write_json(OUTPUT / "paired_feature_outcome_rows.json", rows)
    write_json(OUTPUT / "decision_log.json", decisions)
    write_json(summary_path, summary)
    print(summary_path)
    return summary_path


if __name__ == "__main__":
    run()
