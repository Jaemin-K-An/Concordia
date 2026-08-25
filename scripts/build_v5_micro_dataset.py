#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from concordia.feasibility import (
    BenefitModel,
    RegimeDefinition,
    RegimeProbabilityCalibrator,
    RobustShiftDetector,
    V5SuccessModel,
    V5_FEATURE_SCHEMA,
    build_alignment_case,
    expand_v5_features,
)
from concordia.simulation import SumoAdapter
from run_microscopic_study import _run_one


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/v5/microscopic.yaml"
STUDY = ROOT / "artifacts/studies/v5_micro_calibration"
MODEL_STUDY = ROOT / "artifacts/studies/v5_model_selection"
SHIFT_STUDY = ROOT / "artifacts/studies/v5_shift_detection"


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_network(directory: Path, scenario: str) -> Path:
    scenario_dir = directory / scenario
    scenario_dir.mkdir(parents=True, exist_ok=True)
    nodes = scenario_dir / "corridor.nod.xml"
    edges = scenario_dir / "corridor.edg.xml"
    network = scenario_dir / "corridor.net.xml"
    junction_type = ' type="traffic_light"' if scenario == "signalized" else ""
    nodes.write_text(
        "<nodes>\n"
        '  <node id="s" x="-100" y="0"/><node id="o" x="0" y="0"/>\n'
        '  <node id="m1" x="200" y="0"/><node id="m2" x="400" y="0"/>\n'
        '  <node id="m3" x="600" y="0"/><node id="j" x="800" y="0"'
        f"{junction_type}" + "/>\n"
        '  <node id="d" x="1000" y="0"/><node id="a1" x="250" y="-180"/>\n'
        '  <node id="a2" x="550" y="-180"/>\n'
        "</nodes>\n"
    )
    main_speed = {"merge": 7, "signalized": 12, "two_route": 16, "ring": 9}[scenario]
    alternate_speed = {"merge": 17, "signalized": 16, "two_route": 17, "ring": 13}[scenario]
    alternate_lanes = 2 if scenario == "two_route" else 1
    edges.write_text(
        "<edges>\n"
        '  <edge id="in" from="s" to="o" numLanes="1" speed="20"/>\n'
        '  <edge id="m0" from="o" to="m1" numLanes="1" speed="20"/>\n'
        '  <edge id="m1" from="m1" to="m2" numLanes="1" speed="20"/>\n'
        '  <edge id="m2" from="m2" to="m3" numLanes="1" speed="20"/>\n'
        f'  <edge id="m3" from="m3" to="j" numLanes="1" speed="{main_speed}"/>\n'
        f'  <edge id="a0" from="o" to="a1" numLanes="{alternate_lanes}" speed="{alternate_speed}"/>\n'
        f'  <edge id="a1" from="a1" to="a2" numLanes="{alternate_lanes}" speed="{alternate_speed}"/>\n'
        f'  <edge id="a2" from="a2" to="j" numLanes="{alternate_lanes}" speed="{alternate_speed}"/>\n'
        '  <edge id="out" from="j" to="d" numLanes="1" speed="20"/>\n'
        "</edges>\n"
    )
    binary = SumoAdapter.resolve_binary("netconvert")
    if binary is None:
        raise RuntimeError("netconvert is unavailable")
    subprocess.run(
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
        check=True,
        capture_output=True,
        text=True,
    )
    return network


def _compact(row: dict) -> dict:
    return {
        key: value
        for key, value in row.items()
        if key not in {"state_rows", "phantom_events"}
    }


def _role(seed: int, config: dict) -> str:
    for role, seeds in config["development_roles"].items():
        if seed in seeds:
            return role
    raise RuntimeError(f"micro development seed {seed} has no role")


def run() -> Path:
    existing = STUDY / "raw_metrics.json"
    if (ROOT / "configs/v5/frozen_micro_safety.yaml").is_file():
        if not existing.is_file():
            raise RuntimeError("v5 is frozen but micro development data is missing")
        print(existing)
        return existing
    config = yaml.safe_load(CONFIG.read_text())
    base = yaml.safe_load(
        (ROOT / "configs/experiments/microscopic_policy_matrix.yaml").read_text()
    )
    base.update(
        {
            "vehicle_generation_seconds": config["vehicle_generation_seconds"],
            "maximum_simulation_seconds": config["maximum_simulation_seconds"],
            "preference_epsilon": config["preference_epsilon"],
        }
    )
    regime_package = json.loads((SHIFT_STUDY / "regime_definition.json").read_text())
    regime = RegimeDefinition.from_dict(regime_package["definition"])
    shift_package = json.loads((SHIFT_STUDY / "shift_detector.json").read_text())
    shift = RobustShiftDetector.from_dict(shift_package["detector"])
    calibrated = json.loads((SHIFT_STUDY / "calibrated_model.json").read_text())
    model = V5SuccessModel.from_dict(calibrated["model"])
    calibrator = RegimeProbabilityCalibrator.from_dict(calibrated["calibrator"])
    benefit = BenefitModel.from_dict(calibrated["benefit_model"])
    shift_names = shift_package["feature_names"]
    conditions = [
        (int(seed), str(template[0]), int(template[1]), float(template[2]), str(template[3]))
        for seed in config["development_seeds"]
        for template in config["condition_templates"]
    ]
    analytical = {}
    for seed, scenario, demand, penetration, heterogeneity in conditions:
        case = build_alignment_case(
            scenario=scenario,
            seed=seed,
            demand_scale=demand / 1200.0,
            heterogeneity=heterogeneity,
            navigation_penetration=penetration,
            user_count=6,
            regret_limit=0.08,
            epsilon_grid=[0.0, 0.02, 0.04, 0.08, 0.12, 0.16],
            minimum_relative_ttt_gain=0.01,
            safety_delta=0.25,
            source_split="v5_micro_development_pre_run",
        )
        features = expand_v5_features(case)
        shift_matrix = np.asarray([[features[name] for name in shift_names]])
        dss = float(shift.score(shift_matrix)[0])
        shift_class = shift.classify(shift_matrix)[0]
        features["dss_penetration_interaction"] = dss * penetration
        regime_name = regime.route(features)
        matrix = np.asarray([[features[name] for name in V5_FEATURE_SCHEMA]])
        raw = model.predict_proba(matrix, [regime_name])
        probability = float(calibrator.predict(raw, [regime_name])[0])
        analytical[(seed, scenario, demand, penetration, heterogeneity)] = {
            "case": case,
            "features": features,
            "regime": regime_name,
            "shift_class": shift_class,
            "domain_shift_score": dss,
            "success_probability": probability,
            "predicted_benefit": float(benefit.predict(matrix)[0]),
        }
    with tempfile.TemporaryDirectory(prefix="concordia-v5-micro-dev-") as temporary:
        directory = Path(temporary)
        networks = {
            scenario: _build_network(directory, scenario)
            for scenario in sorted({condition[1] for condition in conditions})
        }
        actual = []
        for seed, scenario, demand, penetration, heterogeneity in conditions:
            for policy in ("B1", "B6"):
                row = _run_one(
                    networks[scenario],
                    base,
                    policy,
                    seed,
                    demand,
                    penetration,
                    heterogeneity,
                )
                row["scenario"] = scenario
                actual.append(_compact(row))
    pair_index = {}
    for row in actual:
        key = (
            row["seed"],
            row["scenario"],
            row["demand_vehicles_per_hour"],
            row["navigation_penetration"],
            row["heterogeneity"],
        )
        pair_index.setdefault(key, {})[row["policy"]] = row
    pairs = []
    for key, values in sorted(pair_index.items()):
        b1, b6 = values["B1"], values["B6"]
        prediction = analytical[key]
        gain = (b1["total_travel_time_seconds"] - b6["total_travel_time_seconds"]) / max(
            b1["total_travel_time_seconds"], 1e-9
        )
        safety_difference = b6["safety"]["cvar_drac_95"] - b1["safety"]["cvar_drac_95"]
        unsafe = safety_difference > 0.25
        success = gain >= 0.01 and not unsafe
        failure_type = (
            "none"
            if success
            else "microscopic_safety_mismatch"
            if unsafe
            else "partial_adoption_feedback"
            if b6["acceptance_rate"] < 0.5
            else "benefit_prediction_error"
        )
        case_id = f"v5-micro-dev-{key[1]}-s{key[0]}-d{key[2]}-p{key[3]:.2f}-{key[4]}"
        micro_features = {
            **prediction["features"],
            "analytical_success_probability": prediction["success_probability"],
            "analytical_benefit": prediction["predicted_benefit"],
        }
        pairs.append(
            {
                "case_id": case_id,
                "seed": key[0],
                "scenario": key[1],
                "demand_vehicles_per_hour": key[2],
                "navigation_penetration": key[3],
                "heterogeneity": key[4],
                "development_role": _role(int(key[0]), config),
                "regime": prediction["regime"],
                "shift_class": prediction["shift_class"],
                "domain_shift_score": prediction["domain_shift_score"],
                "features": prediction["features"],
                "micro_features": micro_features,
                "analytical_probability": prediction["success_probability"],
                "analytical_benefit": prediction["predicted_benefit"],
                "analytical_realized_benefit": prediction["case"]["adaptive_counterfactual"]["relative_ttt_gain"],
                "microscopic_benefit": gain,
                "microscopic_safety_difference": safety_difference,
                "microscopic_safety_violation": int(unsafe),
                "microscopic_success": int(success),
                "failure_type": failure_type,
                "realized_acceptance": b6["acceptance_rate"],
                "b1": b1,
                "b6": b6,
            }
        )
    if len(pairs) != int(config["development_pair_count"]):
        raise RuntimeError("v5 microscopic development pair count mismatch")
    roles = {
        role: sorted(row["case_id"] for row in pairs if row["development_role"] == role)
        for role in config["development_roles"]
    }
    summary = {
        "complete": True,
        "study": "v5 seed-disjoint microscopic development dataset",
        "pair_count": len(pairs),
        "role_counts": dict(Counter(row["development_role"] for row in pairs)),
        "success_count": sum(row["microscopic_success"] for row in pairs),
        "safety_violation_count": sum(
            row["microscopic_safety_violation"] for row in pairs
        ),
        "failure_taxonomy": dict(Counter(row["failure_type"] for row in pairs)),
        "final_holdout_seeds": config["final_holdout_seeds"],
        "final_holdout_materialized": False,
        "source_model_hash": _sha(SHIFT_STUDY / "calibrated_model.json"),
    }
    _write(existing, pairs)
    _write(STUDY / "dataset_summary.json", summary)
    _write(STUDY / "split_manifest.json", {"roles": roles})
    print(existing)
    return existing


if __name__ == "__main__":
    run()
