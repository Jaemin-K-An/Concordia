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
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/concordia-matplotlib-v2")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/concordia-cache-v2")

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from concordia.behavior import AcceptanceModel
from concordia.evaluation import ExperimentRegistry, capture_source_state, paired_comparison
from concordia.models import Route, RouteFeatures
from concordia.populations import generate_population
from concordia.preferences import UtilityModel, preference_slack
from concordia.safety import TrajectoryFrame, parse_sumo_ssm, summarize_safety
from concordia.simulation import SumoAdapter
from concordia.traffic import (
    DetectorObservation,
    LogisticPhantomJamRiskPredictor,
    PHANTOM_FEATURES,
    PhantomJamEventDetector,
    StumpEnsemblePhantomJamRiskPredictor,
    calibration_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiments" / "microscopic_policy_matrix.yaml"
MICRO = ROOT / "artifacts" / "studies" / "microscopic_policy_matrix"
CALIBRATION = ROOT / "artifacts" / "studies" / "phantom_calibration"


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_network(directory: Path) -> Path:
    nodes = directory / "corridor.nod.xml"
    edges = directory / "corridor.edg.xml"
    network = directory / "corridor.net.xml"
    nodes.write_text(
        """<nodes>
  <node id="s" x="-100" y="0"/><node id="o" x="0" y="0"/>
  <node id="m1" x="200" y="0"/><node id="m2" x="400" y="0"/>
  <node id="m3" x="600" y="0"/><node id="j" x="800" y="0"/>
  <node id="d" x="1000" y="0"/><node id="a1" x="250" y="-180"/>
  <node id="a2" x="550" y="-180"/>
</nodes>\n""",
        encoding="utf-8",
    )
    edges.write_text(
        """<edges>
  <edge id="in" from="s" to="o" numLanes="1" speed="20"/>
  <edge id="m0" from="o" to="m1" numLanes="1" speed="20"/>
  <edge id="m1" from="m1" to="m2" numLanes="1" speed="20"/>
  <edge id="m2" from="m2" to="m3" numLanes="1" speed="20"/>
  <edge id="m3" from="m3" to="j" numLanes="1" speed="7"/>
  <edge id="a0" from="o" to="a1" numLanes="1" speed="17"/>
  <edge id="a1" from="a1" to="a2" numLanes="1" speed="17"/>
  <edge id="a2" from="a2" to="j" numLanes="1" speed="17"/>
  <edge id="out" from="j" to="d" numLanes="1" speed="20"/>
</edges>\n""",
        encoding="utf-8",
    )
    netconvert = SumoAdapter.resolve_binary("netconvert")
    if netconvert is None:
        raise RuntimeError("netconvert is unavailable")
    subprocess.run(
        [
            netconvert,
            "--node-files",
            str(nodes),
            "--edge-files",
            str(edges),
            "--output-file",
            str(network),
            "--no-turnarounds",
            "true",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return network


def _write_run_files(
    directory: Path,
    network: Path,
    seed: int,
    demand: int,
    generation_seconds: int,
) -> tuple[Path, Path]:
    route_file = directory / "routes.rou.xml"
    spacing = 3600.0 / demand
    vehicle_count = int(math.floor(generation_seconds / spacing))
    vehicles = "\n".join(
        f'  <vehicle id="v{index:04d}" type="car" route="main" depart="{index * spacing:.3f}"/>'
        for index in range(vehicle_count)
    )
    route_file.write_text(
        """<routes>
  <vType id="car" accel="2.6" decel="4.5" emergencyDecel="8.0" sigma="0.45" tau="1.0" speedDev="0.10">
    <param key="device.ssm.probability" value="1"/>
    <param key="device.ssm.measures" value="TTC PET DRAC"/>
    <param key="device.ssm.thresholds" value="3.0 2.0 3.0"/>
    <param key="device.ssm.file" value="ssm.xml"/>
  </vType>
  <route id="main" edges="in m0 m1 m2 m3 out"/>
  <route id="alternate" edges="in a0 a1 a2 out"/>
"""
        + vehicles
        + "\n</routes>\n",
        encoding="utf-8",
    )
    config = directory / "run.sumocfg"
    config.write_text(
        f"""<configuration>
  <input><net-file value="{network}"/><route-files value="{route_file}"/></input>
  <time><step-length value="1"/></time>
  <processing><collision.action value="warn"/><time-to-teleport value="-1"/></processing>
  <report><no-step-log value="true"/><duration-log.disable value="true"/></report>
  <random_number><seed value="{seed}"/></random_number>
</configuration>\n""",
        encoding="utf-8",
    )
    return config, route_file


def _route_features(adapter: SumoAdapter) -> tuple[Route, Route]:
    main_edges = ("in", "m0", "m1", "m2", "m3", "out")
    alternate_edges = ("in", "a0", "a1", "a2", "out")
    main_eta = sum(float(adapter._traci.edge.getTraveltime(edge)) for edge in main_edges) / 60.0
    alternate_eta = (
        sum(float(adapter._traci.edge.getTraveltime(edge)) for edge in alternate_edges) / 60.0
    )
    main = Route(
        "main",
        tuple(f"m{index}" for index in range(len(main_edges) + 1)),
        RouteFeatures(main_eta, 4.0, 0.0, 0.08, 0.45, 1.0),
    )
    alternate = Route(
        "alternate",
        tuple(f"a{index}" for index in range(len(alternate_edges) + 1)),
        RouteFeatures(alternate_eta, 1.0, 0.0, 0.03, 0.20, 0.2),
    )
    return main, alternate


def _detector(config: dict) -> PhantomJamEventDetector:
    value = config["detector"]
    return PhantomJamEventDetector(
        value["critical_density_vehicles_per_km_per_lane"],
        value["low_speed_threshold_meters_per_second"],
        value["minimum_duration_seconds"],
        value["minimum_amplitude_meters_per_second"],
        value["minimum_absolute_wave_speed_meters_per_second"],
        value["maximum_absolute_wave_speed_meters_per_second"],
        value["minimum_detectors"],
        value["minimum_regression_r_squared"],
        value["ewma_alpha"],
        value["sustained_samples"],
    )


def _run_one(
    network: Path,
    config: dict,
    policy: str,
    seed: int,
    demand: int,
    penetration: float,
    heterogeneity: str,
) -> dict:
    with tempfile.TemporaryDirectory(prefix="concordia-micro-run-") as temporary:
        directory = Path(temporary)
        sumo_config, route_file = _write_run_files(
            directory,
            network,
            seed,
            demand,
            int(config["vehicle_generation_seconds"]),
        )
        binary = SumoAdapter.resolve_binary("sumo")
        if binary is None:
            raise RuntimeError("SUMO is unavailable")
        adapter = SumoAdapter(str(sumo_config), binary=binary)
        rng = random.Random(seed * 1009 + (0 if policy == "B1" else 1))
        expected_vehicles = int(
            math.floor(
                int(config["vehicle_generation_seconds"]) / (3600.0 / demand)
            )
        )
        users = generate_population(
            expected_vehicles,
            "s",
            "d",
            heterogeneity,
            float(config["preference_epsilon"]),
            5.0,
            seed,
        )
        user_by_vehicle = {f"v{index:04d}": user for index, user in enumerate(users)}
        observations = []
        state_rows = []
        frames = []
        departures = {}
        travel_times = []
        lost_time = {}
        offer_count = 0
        accepted_count = 0
        rejected_count = 0
        diverted_count = 0
        navigated_count = 0
        alternate_speeds = []
        utility_model = UtilityModel()
        acceptance_model = AcceptanceModel()
        detector_positions = {"m0": 100.0, "m1": 300.0, "m2": 500.0, "m3": 700.0}
        main_edges = tuple(detector_positions)
        alternate_edges = ("a0", "a1", "a2")
        adapter.start(seed)
        try:
            while (
                adapter._traci.simulation.getMinExpectedNumber() > 0
                and adapter._traci.simulation.getTime()
                < float(config["maximum_simulation_seconds"])
            ):
                snapshot = adapter.step()
                now = snapshot.time
                for vehicle_id in adapter._traci.simulation.getDepartedIDList():
                    departures[vehicle_id] = now
                    if rng.random() > penetration:
                        continue
                    navigated_count += 1
                    main, alternate = _route_features(adapter)
                    if policy == "B1":
                        if alternate.features.time < main.features.time:
                            adapter._traci.vehicle.setRoute(
                                vehicle_id, ["in", "a0", "a1", "a2", "out"]
                            )
                            diverted_count += 1
                        continue
                    user = user_by_vehicle[vehicle_id]
                    utilities = utility_model.utilities(user.preferences, (main, alternate))
                    slack = preference_slack(utilities)["alternate"]
                    if alternate.features.time >= main.features.time or slack > user.epsilon:
                        continue
                    offer_count += 1
                    probability = acceptance_model.probability(
                        slack,
                        utilities["alternate"] - utilities["main"],
                        main.features.time - alternate.features.time,
                        main.features.variability - alternate.features.variability,
                        max(0.0, main.features.time - alternate.features.time),
                    )
                    if rng.random() <= probability:
                        adapter._traci.vehicle.setRoute(
                            vehicle_id, ["in", "a0", "a1", "a2", "out"]
                        )
                        accepted_count += 1
                        diverted_count += 1
                    else:
                        rejected_count += 1
                for vehicle_id in adapter._traci.simulation.getArrivedIDList():
                    if vehicle_id in departures:
                        travel_times.append(now - departures[vehicle_id])
                for vehicle_id in adapter._traci.vehicle.getIDList():
                    lost_time[vehicle_id] = max(
                        lost_time.get(vehicle_id, 0.0),
                        float(adapter._traci.vehicle.getTimeLoss(vehicle_id)),
                    )
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
                selected = [snapshot.edges[edge] for edge in main_edges]
                alternate_speeds.append(
                    float(
                        np.mean(
                            [
                                snapshot.edges[edge].mean_speed_meters_per_second
                                for edge in alternate_edges
                            ]
                        )
                    )
                )
                for edge, position in detector_positions.items():
                    value = snapshot.edges[edge]
                    observations.append(
                        DetectorObservation(
                            now,
                            position,
                            value.density_vehicles_per_km_per_lane,
                            value.mean_speed_meters_per_second,
                        )
                    )
                density = float(np.mean([item.density_vehicles_per_km_per_lane for item in selected]))
                speed = float(np.mean([item.mean_speed_meters_per_second for item in selected]))
                flow = float(np.mean([item.flow_vehicles_per_hour_per_lane for item in selected]))
                saturation = min(2.0, flow / 1800.0)
                state_rows.append(
                    {
                        "time": now,
                        "density": density,
                        "mean_speed": speed,
                        "speed_cv": float(
                            np.mean([item.speed_coefficient_of_variation for item in selected])
                        ),
                        "acceleration_variance": float(
                            np.mean(
                                [
                                    item.acceleration_variance_meters2_per_second4
                                    for item in selected
                                ]
                            )
                        ),
                        "headway_variance": float(
                            np.mean(
                                [
                                    item.headway_variance_seconds2 or 0.0
                                    for item in selected
                                ]
                            )
                        ),
                        "flow": flow,
                        "saturation": saturation,
                        "geometry_complexity": 0.45,
                        "occupancy": float(
                            np.mean([item.occupancy_percent or 0.0 for item in selected])
                        ),
                        "headway_mean": float(
                            np.mean([item.headway_mean_seconds or 0.0 for item in selected])
                        ),
                    }
                )
        finally:
            adapter.close()

        events = _detector(config).detect(observations)
        valid_events = [event for event in events if event.is_valid]
        horizon = float(config["phantom_prediction_horizon_seconds"])
        valid_onsets = [event.start_time for event in valid_events]
        for row in state_rows:
            row["label"] = int(
                any(row["time"] < onset <= row["time"] + horizon for onset in valid_onsets)
            )
        ssm_path = directory / "ssm.xml"
        conflicts = parse_sumo_ssm(str(ssm_path)) if ssm_path.is_file() else []
        safety = summarize_safety(
            frames,
            pet_values=[item.min_pet for item in conflicts if item.min_pet is not None],
        )
        safety_dict = asdict(safety)
        safety_dict.pop("ttc_values")
        safety_dict.pop("drac_values")
        safety_dict.pop("pet_values")
        return {
            "run_id": f"{policy}-{seed}-{demand}-{penetration}-{heterogeneity}",
            "policy": policy,
            "seed": seed,
            "demand_vehicles_per_hour": demand,
            "navigation_penetration": penetration,
            "heterogeneity": heterogeneity,
            "generated_vehicle_count": expected_vehicles,
            "arrived_vehicle_count": len(travel_times),
            "censored_vehicle_count": expected_vehicles - len(travel_times),
            "mean_travel_time_seconds": float(np.mean(travel_times)) if travel_times else None,
            "total_travel_time_seconds": float(np.sum(travel_times)),
            "total_lost_time_seconds": float(sum(lost_time.values())),
            "navigated_vehicle_count": navigated_count,
            "offer_count": offer_count,
            "accepted_count": accepted_count,
            "rejected_count": rejected_count,
            "acceptance_rate": accepted_count / max(1, offer_count),
            "diverted_vehicle_count": diverted_count,
            "diverted_fraction": diverted_count / max(1, expected_vehicles),
            "valid_phantom_jam": bool(valid_events),
            "valid_phantom_event_count": len(valid_events),
            "phantom_candidate_count": len(events),
            "phantom_events": [asdict(event) for event in events],
            "safety": safety_dict,
            "ssm_conflict_count": len(conflicts),
            "state_rows": state_rows[::5],
            "network_hash": _checksum(network),
            "route_file_hash": _checksum(route_file),
            "synthetic": True,
            "alternate_mean_speed_meters_per_second": float(np.mean(alternate_speeds)),
        }


def _calibration(rows: list[dict], config: dict) -> dict:
    samples = []
    for row in rows:
        if row["policy"] != "B1":
            continue
        for state in row["state_rows"]:
            samples.append(
                {
                    **state,
                    "run_id": row["run_id"],
                    "seed": row["seed"],
                    "demand_vehicles_per_hour": row["demand_vehicles_per_hour"],
                }
            )
    train = [sample for sample in samples if sample["seed"] != max(config["seeds"])]
    test = [sample for sample in samples if sample["seed"] == max(config["seeds"])]
    feature_names = list(PHANTOM_FEATURES)

    def arrays(data):
        return (
            np.asarray([[row[name] for name in feature_names] for row in data], dtype=float),
            np.asarray([row["label"] for row in data], dtype=int),
        )

    result = {
        "complete": False,
        "split_unit": ["simulation_run", "seed", "scenario_condition"],
        "held_out_seed": max(config["seeds"]),
        "train_run_count": len(set(row["run_id"] for row in train)),
        "test_run_count": len(set(row["run_id"] for row in test)),
        "train_sample_count": len(train),
        "test_sample_count": len(test),
        "positive_labels": sum(row["label"] for row in samples),
        "models": {},
    }
    if (
        train
        and test
        and len({row["label"] for row in train}) == 2
        and len({row["label"] for row in test}) == 2
    ):
        train_x, train_y = arrays(train)
        test_x, test_y = arrays(test)
        models = (
            ("logistic_regression", LogisticPhantomJamRiskPredictor(iterations=800)),
            ("calibrated_stump_ensemble", StumpEnsemblePhantomJamRiskPredictor()),
        )
        for name, model in models:
            model.fit(train_x, train_y)
            probabilities = model.predict_proba(test_x)
            calibration_curve = []
            for lower, upper in zip(np.linspace(0, 0.9, 10), np.linspace(0.1, 1.0, 10)):
                members = (probabilities >= lower) & (
                    probabilities <= upper if upper == 1.0 else probabilities < upper
                )
                if members.any():
                    calibration_curve.append(
                        {
                            "mean_predicted_probability": float(probabilities[members].mean()),
                            "observed_event_rate": float(test_y[members].mean()),
                            "count": int(members.sum()),
                        }
                    )
            result["models"][name] = {
                "metrics": asdict(calibration_metrics(test_y, probabilities)),
                "model_card": model.model_card(),
                "calibration_curve": calibration_curve,
            }
        result["complete"] = True
        result["selected_model"] = min(
            result["models"],
            key=lambda name: result["models"][name]["metrics"]["brier_score"],
        )
    else:
        result["reason"] = "held-out run split did not contain both VALID-event classes"
    return {"summary": result, "dataset": samples}


def _statistics(rows: list[dict], config: dict) -> dict:
    paired = defaultdict(dict)
    for row in rows:
        key = (
            row["seed"],
            row["demand_vehicles_per_hour"],
            row["navigation_penetration"],
            row["heterogeneity"],
        )
        paired[key][row["policy"]] = row
    complete_pairs = [value for value in paired.values() if set(value) == {"B1", "B6"}]
    b1_binary = np.asarray([int(pair["B1"]["valid_phantom_jam"]) for pair in complete_pairs])
    b6_binary = np.asarray([int(pair["B6"]["valid_phantom_jam"]) for pair in complete_pairs])
    discordant_b1_only = int(np.sum((b1_binary == 1) & (b6_binary == 0)))
    discordant_b6_only = int(np.sum((b1_binary == 0) & (b6_binary == 1)))
    try:
        from scipy.stats import binomtest, spearmanr

        mcnemar_p = (
            float(
                binomtest(
                    min(discordant_b1_only, discordant_b6_only),
                    discordant_b1_only + discordant_b6_only,
                    0.5,
                ).pvalue
            )
            if discordant_b1_only + discordant_b6_only
            else 1.0
        )
    except ImportError:
        mcnemar_p = None
        spearmanr = None
    b1_duration = [
        sum(event["validation"]["duration_seconds"] for event in pair["B1"]["phantom_events"] if event["validation"]["status"] == "VALID")
        for pair in complete_pairs
    ]
    b6_duration = [
        sum(event["validation"]["duration_seconds"] for event in pair["B6"]["phantom_events"] if event["validation"]["status"] == "VALID")
        for pair in complete_pairs
    ]
    b1_drac = [pair["B1"]["safety"]["cvar_drac_95"] for pair in complete_pairs]
    b6_drac = [pair["B6"]["safety"]["cvar_drac_95"] for pair in complete_pairs]
    rng = np.random.default_rng(43)
    differences = np.asarray(b6_drac) - np.asarray(b1_drac)
    draws = differences[
        rng.integers(0, len(differences), size=(5000, len(differences)))
    ].mean(axis=1)
    upper = float(np.percentile(draws, 97.5))
    margin = float(config["safety"]["noninferiority_margin"])

    safety_features = np.asarray(
        [
            [
                row["demand_vehicles_per_hour"] / 1800.0,
                row["navigation_penetration"],
                row["diverted_fraction"],
                row["total_lost_time_seconds"] / max(1, row["generated_vehicle_count"]),
            ]
            for row in rows
        ],
        dtype=float,
    )
    safety_target = np.asarray([row["safety"]["cvar_drac_95"] for row in rows], dtype=float)
    train_mask = np.asarray([row["seed"] != max(config["seeds"]) for row in rows])
    design = np.column_stack([np.ones(train_mask.sum()), safety_features[train_mask]])
    coefficients, *_ = np.linalg.lstsq(design, safety_target[train_mask], rcond=None)
    prediction = np.column_stack(
        [np.ones((~train_mask).sum()), safety_features[~train_mask]]
    ) @ coefficients
    realized = safety_target[~train_mask]
    if spearmanr is None or len(realized) < 2:
        rank_correlation = None
    else:
        correlation = spearmanr(prediction, realized).statistic
        rank_correlation = None if not np.isfinite(correlation) else float(correlation)
    predicted_threshold = float(np.median(prediction)) if len(prediction) else 0.0
    realized_threshold = float(np.percentile(realized, 75)) if len(realized) else 0.0
    false_safe = int(np.sum((prediction <= predicted_threshold) & (realized >= realized_threshold)))
    return {
        "matched_pair_count": len(complete_pairs),
        "H3": {
            "primary_metric": "P(VALID phantom jam)",
            "B1_probability": float(b1_binary.mean()),
            "B6_probability": float(b6_binary.mean()),
            "paired_probability_difference_B1_minus_B6": float(
                (b1_binary - b6_binary).mean()
            ),
            "discordant_B1_only": discordant_b1_only,
            "discordant_B6_only": discordant_b6_only,
            "exact_mcnemar_p": mcnemar_p,
            "duration_paired": paired_comparison(b1_duration, b6_duration, seed=47),
        },
        "H4": {
            "primary_metric": config["safety"]["primary_metric"],
            "noninferiority_margin": margin,
            "paired_mean_difference_B6_minus_B1": float(differences.mean()),
            "bootstrap_ci95": [
                float(np.percentile(draws, 2.5)),
                upper,
            ],
            "noninferior": upper <= margin,
            "claim_boundary": "microscopic surrogate safety only; not crash probability",
        },
        "safety_predictor_calibration": {
            "model": "held-out-seed linear surrogate",
            "coefficients": coefficients.tolist(),
            "rank_correlation": rank_correlation,
            "false_safe_count": false_safe,
            "held_out_count": int(len(realized)),
            "predicted_values": prediction.tolist(),
            "realized_values": realized.tolist(),
        },
    }


def _figures(rows: list[dict], statistics: dict) -> list[Path]:
    directory = MICRO / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    outputs = []
    fig, axis = plt.subplots(figsize=(6.2, 4.2))
    h3 = statistics["H3"]
    axis.bar(["B1", "B6"], [h3["B1_probability"], h3["B6_probability"]], color=["#777777", "#111111"])
    axis.set_ylim(0, 1)
    axis.set_ylabel("P(VALID phantom jam)")
    fig.tight_layout()
    path = directory / "b1_b6_phantom_probability.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)

    for name, field, ylabel in (
        ("phantom_event_duration", "duration_seconds", "VALID event duration (s)"),
        ("phantom_event_amplitude", "oscillation_amplitude_meters_per_second", "VALID amplitude (m/s)"),
    ):
        fig, axis = plt.subplots(figsize=(6.2, 4.2))
        values = {
            policy: [
                event["validation"][field]
                for row in rows
                if row["policy"] == policy
                for event in row["phantom_events"]
                if event["validation"]["status"] == "VALID"
            ]
            for policy in ("B1", "B6")
        }
        axis.boxplot(
            [values["B1"] or [0], values["B6"] or [0]],
            tick_labels=["B1", "B6"],
        )
        axis.set_ylabel(ylabel)
        fig.tight_layout()
        path = directory / f"{name}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        outputs.append(path)

    fig, axis = plt.subplots(figsize=(6.2, 4.2))
    axis.boxplot(
        [
            [row["safety"]["cvar_drac_95"] for row in rows if row["policy"] == policy]
            for policy in ("B1", "B6")
        ],
        tick_labels=["B1", "B6"],
    )
    axis.set_ylabel("CVaR95 DRAC (m/s²)")
    fig.tight_layout()
    path = directory / "safety_cvar_b1_b6.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)
    calibration = statistics["safety_predictor_calibration"]
    fig, axis = plt.subplots(figsize=(5.2, 5.2))
    axis.scatter(
        calibration["predicted_values"],
        calibration["realized_values"],
        color="#111111",
    )
    limit = max(
        calibration["predicted_values"] + calibration["realized_values"] + [1.0]
    )
    axis.plot([0, limit], [0, limit], color="#999999", linestyle="--")
    axis.set_xlabel("Predicted DRAC CVaR")
    axis.set_ylabel("Realized DRAC CVaR")
    axis.set_title("Held-out seed safety calibration")
    fig.tight_layout()
    path = directory / "predicted_vs_realized_safety.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)
    return outputs


def _calibration_figure(calibration: dict) -> Path:
    directory = CALIBRATION / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(5.2, 5.2))
    if calibration.get("complete"):
        for name, model in calibration["models"].items():
            curve = model["calibration_curve"]
            axis.plot(
                [item["mean_predicted_probability"] for item in curve],
                [item["observed_event_rate"] for item in curve],
                marker="o",
                label=name,
            )
        axis.plot([0, 1], [0, 1], color="#999999", linestyle="--")
        axis.legend(fontsize=7)
    else:
        axis.text(0.5, 0.5, "NOT CALIBRATED\ninsufficient held-out classes", ha="center")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Predicted probability")
    axis.set_ylabel("Observed VALID-event rate")
    axis.set_title("Phantom predictor calibration")
    fig.tight_layout()
    path = directory / "calibration_curve.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _materialize_processed_contract() -> list[Path]:
    """Expose canonical raw/processed names without changing recorded evidence."""
    micro_processed = MICRO / "processed_metrics.json"
    calibration_raw = CALIBRATION / "raw_metrics.json"
    calibration_processed = CALIBRATION / "processed_metrics.json"
    shutil.copyfile(MICRO / "statistical_tests.json", micro_processed)
    shutil.copyfile(CALIBRATION / "dataset.json", calibration_raw)
    shutil.copyfile(CALIBRATION / "summary.json", calibration_processed)
    return [micro_processed, calibration_raw, calibration_processed]


def run() -> Path:
    source_commit, source_dirty = capture_source_state()
    started = datetime.now(timezone.utc)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="concordia-micro-network-") as temporary:
        network = _build_network(Path(temporary))
        rows = []
        for seed in config["seeds"]:
            for demand in config["demand_vehicles_per_hour"]:
                for penetration in config["navigation_penetration"]:
                    for heterogeneity in config["heterogeneity"]:
                        for policy in config["policies"]:
                            rows.append(
                                _run_one(
                                    network,
                                    config,
                                    policy,
                                    int(seed),
                                    int(demand),
                                    float(penetration),
                                    str(heterogeneity),
                                )
                            )
    calibration = _calibration(rows, config)
    statistics = _statistics(rows, config)
    MICRO.mkdir(parents=True, exist_ok=True)
    raw_rows = [{key: value for key, value in row.items() if key != "state_rows"} for row in rows]
    _write_json(MICRO / "raw_metrics.json", raw_rows)
    _write_json(MICRO / "statistical_tests.json", statistics)
    figures = _figures(raw_rows, statistics)
    summary = {
        "complete": True,
        "study": "Study II — Microscopic Phantom/Safety",
        "simulator_version": SumoAdapter.simulator_version(),
        "run_count": len(rows),
        "matched_pair_count": statistics["matched_pair_count"],
        "H3": statistics["H3"],
        "H4": statistics["H4"],
        "claim_boundary": config["claim_boundary"],
    }
    _write_json(MICRO / "summary.json", summary)
    _write_json(CALIBRATION / "dataset.json", calibration["dataset"])
    _write_json(CALIBRATION / "summary.json", calibration["summary"])
    _write_json(CALIBRATION / "statistical_tests.json", calibration["summary"].get("models", {}))
    calibration_figure = _calibration_figure(calibration["summary"])
    contract_outputs = _materialize_processed_contract()
    ended = datetime.now(timezone.utc)
    outputs = [
        MICRO / "raw_metrics.json",
        MICRO / "statistical_tests.json",
        MICRO / "summary.json",
        CALIBRATION / "dataset.json",
        CALIBRATION / "summary.json",
        CALIBRATION / "statistical_tests.json",
        calibration_figure,
        *contract_outputs,
        *figures,
    ]
    run_dir = ExperimentRegistry(str(ROOT / "artifacts" / "runs")).create(
        config,
        summary,
        simulator_version=summary["simulator_version"],
        input_paths=(str(CONFIG.relative_to(ROOT)),),
        external_output_paths=tuple(str(path.relative_to(ROOT)) for path in outputs),
        started_at=started,
        ended_at=ended,
        source_commit=source_commit,
        source_dirty=source_dirty,
    )
    shutil.copyfile(run_dir / "manifest.json", MICRO / "manifest.json")
    shutil.copyfile(run_dir / "manifest.json", CALIBRATION / "manifest.json")
    print(MICRO / "summary.json")
    return MICRO / "summary.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-if-valid", action="store_true")
    arguments = parser.parse_args()
    existing = MICRO / "summary.json"
    if arguments.reuse_if_valid and existing.is_file() and json.loads(
        existing.read_text(encoding="utf-8")
    ).get("complete"):
        _materialize_processed_contract()
        print(existing)
    else:
        run()
