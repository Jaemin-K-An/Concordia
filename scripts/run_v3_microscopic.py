#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/concordia-matplotlib-v3")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/concordia-cache-v3")

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from concordia.evaluation import (
    ExperimentRegistry,
    capture_source_state,
    paired_comparison,
    summarize_selective_policy,
)
from concordia.feasibility import BootstrapFeasibilityEnsemble, FEATURE_SCHEMA, FeasibilityGate
from concordia.feasibility.dataset import build_alignment_case
from concordia.selective import SelectiveInterventionPolicy
from concordia.simulation import SumoAdapter
from concordia.traffic import (
    PHANTOM_FEATURES,
    LogisticPhantomJamRiskPredictor,
    StumpEnsemblePhantomJamRiskPredictor,
    calibration_metrics,
)
from run_microscopic_study import _build_network, _run_one


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/v3/microscopic.yaml"
BASE_CONFIG = ROOT / "configs/experiments/microscopic_policy_matrix.yaml"
MODEL_STUDY = ROOT / "artifacts/studies/v3_feasibility_prediction"
OUTPUT = ROOT / "artifacts/studies/v3_microscopic_selective"
V2_MICRO = ROOT / "artifacts/studies/microscopic_policy_matrix/raw_metrics.json"


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _regression_predict(package: dict, matrix: np.ndarray) -> np.ndarray:
    mean = np.asarray(package["mean"], dtype=float)
    scale = np.asarray(package["scale"], dtype=float)
    coefficients = np.asarray(package["coefficients"], dtype=float)
    return package["intercept"] + ((matrix - mean) / scale) @ coefficients


def _phantom_calibration(rows: list[dict]) -> tuple[dict, list[dict]]:
    run_samples = {}
    for row in rows:
        samples = [
            {
                **state,
                "run_id": row["run_id"],
                "seed": row["seed"],
                "demand_vehicles_per_hour": row["demand_vehicles_per_hour"],
            }
            for state in row["state_rows"]
        ]
        run_samples[row["run_id"]] = samples
    by_class = defaultdict(list)
    for run_id, samples in run_samples.items():
        by_class[int(any(sample["label"] for sample in samples))].append(run_id)
    for values in by_class.values():
        values.sort()
    result = {
        "complete": False,
        "split_unit": "simulation_run",
        "split_strategy": "run-event-stratified deterministic split",
        "positive_run_count": len(by_class[1]),
        "negative_run_count": len(by_class[0]),
        "models": {},
        "phantom_gate_used": False,
    }
    if len(by_class[0]) < 2 or len(by_class[1]) < 2:
        result["reason"] = "fewer than two positive or negative simulation runs"
        return result, [sample for values in run_samples.values() for sample in values]
    test_runs = {by_class[label][-1] for label in (0, 1)}
    train_runs = set(run_samples) - test_runs
    train = [sample for run_id in sorted(train_runs) for sample in run_samples[run_id]]
    test = [sample for run_id in sorted(test_runs) for sample in run_samples[run_id]]

    def arrays(values: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray([[row[name] for name in PHANTOM_FEATURES] for row in values], dtype=float),
            np.asarray([row["label"] for row in values], dtype=int),
        )

    train_x, train_y = arrays(train)
    test_x, test_y = arrays(test)
    result.update(
        {
            "train_run_ids": sorted(train_runs),
            "test_run_ids": sorted(test_runs),
            "train_sample_count": len(train),
            "test_sample_count": len(test),
            "train_positive_count": int(train_y.sum()),
            "test_positive_count": int(test_y.sum()),
        }
    )
    if len(np.unique(train_y)) < 2 or len(np.unique(test_y)) < 2:
        result["reason"] = "run-level split lacks both state-label classes in one partition"
        return result, train + test
    models = (
        ("logistic_regression", LogisticPhantomJamRiskPredictor(iterations=1000)),
        ("calibrated_stump_ensemble", StumpEnsemblePhantomJamRiskPredictor()),
    )
    for name, model in models:
        model.fit(train_x, train_y)
        probabilities = model.predict_proba(test_x)
        curve = []
        for lower in np.linspace(0.0, 0.9, 10):
            upper = lower + 0.1
            members = (probabilities >= lower) & (
                probabilities <= upper if upper >= 1.0 else probabilities < upper
            )
            if members.any():
                curve.append(
                    {
                        "mean_predicted_probability": float(probabilities[members].mean()),
                        "observed_event_rate": float(test_y[members].mean()),
                        "count": int(members.sum()),
                    }
                )
        result["models"][name] = {
            "metrics": asdict(calibration_metrics(test_y, probabilities)),
            "model_card": model.model_card(),
            "calibration_curve": curve,
        }
    result["complete"] = True
    result["selected_model"] = min(
        result["models"],
        key=lambda name: result["models"][name]["metrics"]["brier_score"],
    )
    result["phantom_gate_used"] = True
    return result, train + test


def _fit_v2_safety_difference() -> dict:
    rows = json.loads(V2_MICRO.read_text(encoding="utf-8"))
    pairs = defaultdict(dict)
    for row in rows:
        key = (
            row["seed"],
            row["demand_vehicles_per_hour"],
            row["navigation_penetration"],
            row["heterogeneity"],
        )
        pairs[key][row["policy"]] = row
    matrix = []
    target = []
    for key, pair in pairs.items():
        if set(pair) != {"B1", "B6"}:
            continue
        _, demand, penetration, heterogeneity = key
        matrix.append([1.0, float(demand) / 1800.0, float(penetration), heterogeneity == "high"])
        target.append(pair["B6"]["safety"]["cvar_drac_95"] - pair["B1"]["safety"]["cvar_drac_95"])
    x = np.asarray(matrix, dtype=float)
    y = np.asarray(target, dtype=float)
    penalty = np.eye(x.shape[1]) * 1e-3
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(x.T @ x + penalty, x.T @ y)
    residual = y - x @ coefficients
    return {
        "coefficients": coefficients.tolist(),
        "residual_standard_deviation": float(np.std(residual, ddof=1)),
        "source": "v2 microscopic development evidence only",
        "pair_count": len(y),
    }


def _safety_upper(model: dict, demand: int) -> float:
    values = np.asarray([1.0, demand / 1800.0, 1.0, 1.0])
    return float(
        values @ np.asarray(model["coefficients"])
        + 1.645 * model["residual_standard_deviation"]
    )


def _figures(rows: list[dict], metrics: dict, calibration: dict) -> list[Path]:
    directory = OUTPUT / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    outputs = []
    fig, axis = plt.subplots(figsize=(6.4, 4.2))
    policies = ["B1", "B6", "V3"]
    values = [
        np.mean([int(row["valid_phantom_jam"]) for row in rows if row["policy"] == policy])
        for policy in policies
    ]
    axis.bar(policies, values, color=["#888888", "#333333", "#0b6e4f"])
    axis.set(ylim=(0, 1), ylabel="P(VALID phantom jam)")
    fig.tight_layout()
    path = directory / "phantom_probability_b1_b6_v3.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)
    fig, axis = plt.subplots(figsize=(5.4, 5.0))
    if calibration.get("complete"):
        for name, model in calibration["models"].items():
            curve = model["calibration_curve"]
            axis.plot(
                [item["mean_predicted_probability"] for item in curve],
                [item["observed_event_rate"] for item in curve],
                marker="o",
                label=name,
            )
        axis.legend(fontsize=7)
    else:
        axis.text(0.5, 0.5, "PHANTOM GATE EXCLUDED", ha="center")
    axis.plot([0, 1], [0, 1], linestyle="--", color="#999999")
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted probability", ylabel="Observed rate")
    fig.tight_layout()
    path = directory / "run_group_calibration.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)
    fig, axis = plt.subplots(figsize=(6.4, 4.2))
    primary = metrics["V3"]
    axis.bar(
        ["precision", "coverage", "population benefit"],
        [
            primary["intervention_precision"],
            primary["coverage"],
            primary["population_benefit_rate"],
        ],
        color="#0b6e4f",
    )
    axis.set_ylim(0, 1)
    fig.tight_layout()
    path = directory / "microscopic_selectivity.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)
    return outputs


def run() -> Path:
    source_commit, source_dirty = capture_source_state()
    started = datetime.now(timezone.utc)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    base = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    frozen_path = ROOT / "configs/v3/frozen_thresholds.yaml"
    selected_path = MODEL_STUDY / "selected_model.json"
    frozen = yaml.safe_load(frozen_path.read_text(encoding="utf-8"))
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    if _sha(selected_path) != frozen["selected_model_hash"]:
        raise RuntimeError("selected feasibility model changed after freeze")
    with tempfile.TemporaryDirectory(prefix="concordia-v3-micro-") as temporary:
        network = _build_network(Path(temporary))
        baseline_rows = [
            _run_one(
                network,
                base,
                "B1",
                int(seed),
                int(demand),
                float(config["navigation_penetration"]),
                str(config["heterogeneity"]),
            )
            for seed in config["seeds"]
            for demand in config["metastable_search_demand"]
        ]
        calibration, calibration_dataset = _phantom_calibration(baseline_rows)
        pair_rows = []
        baseline_by_key = {
            (row["seed"], row["demand_vehicles_per_hour"]): row for row in baseline_rows
        }
        for seed in config["seeds"]:
            for demand in config["selective_test_demand"]:
                pair_rows.append(baseline_by_key[(seed, demand)])
                pair_rows.append(
                    _run_one(
                        network,
                        base,
                        "B6",
                        int(seed),
                        int(demand),
                        float(config["navigation_penetration"]),
                        str(config["heterogeneity"]),
                    )
                )
    pairs = defaultdict(dict)
    for row in pair_rows:
        pairs[(row["seed"], row["demand_vehicles_per_hour"])][row["policy"]] = row
    ensemble = BootstrapFeasibilityEnsemble.from_dict(selected["ensemble"])
    safety_model = _fit_v2_safety_difference()
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
    decision_log = []
    policy_rows = {"B6": [], "V3": []}
    selected_rows = []
    for (seed, demand), pair in sorted(pairs.items()):
        analytical = build_alignment_case(
            scenario="merge",
            seed=int(seed),
            demand_scale=float(demand) / 1200.0,
            heterogeneity=str(config["heterogeneity"]),
            navigation_penetration=float(config["navigation_penetration"]),
            user_count=6,
            regret_limit=0.08,
            epsilon_grid=[0.0, 0.02, 0.04, 0.08, 0.12, 0.16],
            minimum_relative_ttt_gain=float(frozen["minimum_relative_ttt_gain"]),
            safety_delta=float(frozen["safety_delta"]),
            source_split="microscopic_transfer",
        )
        matrix = np.asarray([[analytical["features"][name] for name in FEATURE_SCHEMA]])
        probability, uncertainty, lower = ensemble.predict(matrix)
        benefit = _regression_predict(selected["benefit_regression"], matrix)[0]
        benefit_lcb = benefit - frozen["benefit_lcb_z"] * selected["benefit_regression"][
            "residual_standard_deviation"
        ]
        safety_upper = _safety_upper(safety_model, int(demand))
        decision = policy.decide(
            case_id=f"micro-s{seed}-d{demand}",
            p_win=float(probability[0]),
            p_win_lower=float(lower[0]),
            uncertainty=float(uncertainty[0]),
            alignment_potential=analytical["features"]["alignment_potential_score"],
            route_overlap=analytical["features"]["route_overlap"],
            safety_upper_difference=safety_upper,
            acceptance_probability=analytical["features"]["acceptance_probability"],
            ttt_lcb_gain=float(benefit_lcb),
            predicted_tail_loss=max(0.0, -float(benefit_lcb)),
            legal=True,
        )
        b1 = pair["B1"]
        b6 = pair["B6"]
        relative_gain = (b1["total_travel_time_seconds"] - b6["total_travel_time_seconds"]) / max(
            b1["total_travel_time_seconds"], 1e-9
        )
        safety_difference = b6["safety"]["cvar_drac_95"] - b1["safety"]["cvar_drac_95"]
        counterfactual_success = (
            relative_gain >= frozen["minimum_relative_ttt_gain"]
            and safety_difference <= frozen["safety_delta"]
        )
        selected_row = b6 if decision.intervene else b1
        selected_rows.append({**selected_row, "policy": "V3"})
        for policy_name, intervene in (("B6", True), ("V3", decision.intervene)):
            chosen_ttt = b6["total_travel_time_seconds"] if intervene else b1["total_travel_time_seconds"]
            policy_rows[policy_name].append(
                {
                    "case_id": decision.case_id,
                    "intervene": intervene,
                    "success": bool(intervene and counterfactual_success),
                    "counterfactual_success": counterfactual_success,
                    "system_ttt_gain": b1["total_travel_time_seconds"] - chosen_ttt,
                    "baseline_ttt": b1["total_travel_time_seconds"],
                    "selected_ttt": chosen_ttt,
                    "regret_violation": False,
                    "safety_violation": safety_difference > frozen["safety_delta"],
                    "legal_violation": False,
                }
            )
        decision_log.append(
            {
                **decision.__dict__,
                "outcome": "SUCCESS" if decision.intervene and counterfactual_success else "FAILURE" if decision.intervene else "ABSTAIN",
                "relative_ttt_gain_realized": relative_gain,
                "safety_difference_realized": safety_difference,
                "safety_upper_predicted": safety_upper,
                "threshold_hash": _sha(frozen_path),
                "phantom_gate_used": calibration["phantom_gate_used"],
            }
        )
    metrics = {name: summarize_selective_policy(values) for name, values in policy_rows.items()}
    all_policy_rows = pair_rows + selected_rows
    phantom_probability = {
        policy_name: float(
            np.mean([int(row["valid_phantom_jam"]) for row in all_policy_rows if row["policy"] == policy_name])
        )
        for policy_name in ("B1", "B6", "V3")
    }
    b6_differences = []
    v3_differences = []
    for index, ((_seed, _demand), pair) in enumerate(sorted(pairs.items())):
        difference = pair["B6"]["safety"]["cvar_drac_95"] - pair["B1"]["safety"]["cvar_drac_95"]
        b6_differences.append(difference)
        v3_differences.append(difference if policy_rows["V3"][index]["intervene"] else 0.0)
    statistics = {
        "H3_phantom_probability": phantom_probability,
        "H3_paired_B1_vs_V3": paired_comparison(
            [int(row["valid_phantom_jam"]) for row in all_policy_rows if row["policy"] == "B1"],
            [int(row["valid_phantom_jam"]) for row in all_policy_rows if row["policy"] == "V3"],
            seed=127,
        ),
        "H11_B6_safety_failure_count": int(sum(value > frozen["safety_delta"] for value in b6_differences)),
        "H11_V3_safety_failure_count": int(sum(value > frozen["safety_delta"] for value in v3_differences)),
        "H11_correctly_abstained_risky_count": int(
            sum(
                value > frozen["safety_delta"] and not policy_rows["V3"][index]["intervene"]
                for index, value in enumerate(b6_differences)
            )
        ),
        "B1_vs_V3_TTT": paired_comparison(
            [row["baseline_ttt"] for row in policy_rows["V3"]],
            [row["selected_ttt"] for row in policy_rows["V3"]],
            seed=131,
        ),
    }
    summary = {
        "complete": True,
        "study": "Study VII — Microscopic Safety-Selectivity",
        "simulator_version": SumoAdapter.simulator_version(),
        "metastable_search_run_count": len(baseline_rows),
        "selective_pair_count": len(pairs),
        "policy_metrics": metrics,
        "phantom_calibration_complete": calibration["complete"],
        "phantom_gate_used": calibration["phantom_gate_used"],
        "statistics": statistics,
        "claim_boundary": config["claim_boundary"],
    }
    raw = [
        {key: value for key, value in row.items() if key != "state_rows"}
        for row in baseline_rows
        + [row for row in pair_rows if row["policy"] == "B6"]
        + selected_rows
    ]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    raw_path = OUTPUT / "raw_metrics.json"
    processed_path = OUTPUT / "processed_metrics.json"
    statistics_path = OUTPUT / "statistical_tests.json"
    summary_path = OUTPUT / "summary.json"
    calibration_path = OUTPUT / "phantom_calibration.json"
    calibration_data_path = OUTPUT / "phantom_calibration_dataset.json"
    decision_path = OUTPUT / "decision_log.json"
    _write(raw_path, raw)
    _write(processed_path, metrics)
    _write(statistics_path, statistics)
    _write(summary_path, summary)
    _write(calibration_path, calibration)
    _write(calibration_data_path, calibration_dataset)
    _write(decision_path, decision_log)
    figures = _figures(all_policy_rows, metrics, calibration)
    ended = datetime.now(timezone.utc)
    outputs = (
        raw_path,
        processed_path,
        statistics_path,
        summary_path,
        calibration_path,
        calibration_data_path,
        decision_path,
        *figures,
    )
    run_dir = ExperimentRegistry(str(ROOT / "artifacts/runs")).create(
        config,
        summary,
        simulator_version=SumoAdapter.simulator_version(),
        input_paths=(
            str(CONFIG.relative_to(ROOT)),
            "configs/v3/frozen_thresholds.yaml",
            str(selected_path.relative_to(ROOT)),
            str(V2_MICRO.relative_to(ROOT)),
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
