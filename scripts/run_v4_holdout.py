#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/concordia-matplotlib-v4")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/concordia-cache-v4")

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from concordia.evaluation import ExperimentRegistry, capture_source_state, summarize_selective_policy
from concordia.feasibility import (
    BootstrapFeasibilityEnsemble,
    FEATURE_SCHEMA,
    V4PredictionBundle,
    V4_FEATURE_SCHEMA,
    build_alignment_case,
    expand_v4_features,
    group_metrics,
)
from concordia.selective import PrecisionConstrainedPolicy, V4DecisionInputs


ROOT = Path(__file__).resolve().parents[1]
MODEL_STUDY = ROOT / "artifacts/studies/v4_model_selection"
VALIDATION = ROOT / "artifacts/studies/v4_precision_validation"
OUTPUT = ROOT / "artifacts/studies/v4_frozen_holdout"
V3_STUDY = ROOT / "artifacts/studies/v3_feasibility_prediction"


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _regression_predict(package: dict, matrix: np.ndarray) -> np.ndarray:
    mean = np.asarray(package["mean"])
    scale = np.asarray(package["scale"])
    coefficients = np.asarray(package["coefficients"])
    return package["intercept"] + ((matrix - mean) / scale) @ coefficients


def _extended_metrics(rows: list[dict]) -> dict:
    summary = summarize_selective_policy(rows)
    interventions = [row for row in rows if row["intervene"]]
    relative = [row["relative_ttt_gain"] for row in interventions]
    summary.update(
        {
            "median_relative_ttt_gain": float(np.median(relative)) if relative else 0.0,
            "p10_relative_ttt_gain": float(np.percentile(relative, 10)) if relative else 0.0,
            "maximum_regret": max((row["maximum_regret"] for row in interventions), default=0.0),
            "mean_decision_latency_seconds": float(
                np.mean([row["decision_latency_seconds"] for row in rows])
            ),
            "p95_decision_latency_seconds": float(
                np.percentile([row["decision_latency_seconds"] for row in rows], 95)
            ),
        }
    )
    return summary


def _v4_policy(selection: dict, name: str) -> PrecisionConstrainedPolicy:
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


def run() -> Path:
    source_commit, source_dirty = capture_source_state()
    started = datetime.now(timezone.utc)
    split = yaml.safe_load((ROOT / "configs/v4/splits.yaml").read_text(encoding="utf-8"))
    prereg = yaml.safe_load(
        (ROOT / "configs/v4/preregistration.yaml").read_text(encoding="utf-8")
    )
    model_config = yaml.safe_load(
        (ROOT / "configs/v4/model_selection.yaml").read_text(encoding="utf-8")
    )
    frozen_model_path = ROOT / model_config["freeze_model"]
    frozen_threshold_path = ROOT / model_config["freeze_thresholds"]
    manifest_path = ROOT / model_config["freeze_manifest"]
    if not all(path.is_file() for path in (frozen_model_path, frozen_threshold_path, manifest_path)):
        raise RuntimeError("v4 holdout is forbidden before model and threshold freeze")
    model_hash_before = _sha(frozen_model_path)
    threshold_hash_before = _sha(frozen_threshold_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if model_hash_before != manifest["frozen_model_hash"]:
        raise RuntimeError("frozen v4 model checksum mismatch")
    if threshold_hash_before != manifest["frozen_threshold_hash"]:
        raise RuntimeError("frozen v4 threshold checksum mismatch")
    frozen_model = yaml.safe_load(frozen_model_path.read_text(encoding="utf-8"))
    frozen = yaml.safe_load(frozen_threshold_path.read_text(encoding="utf-8"))
    for name, relative in frozen_model["artifact_paths"].items():
        path = ROOT / relative
        if _sha(path) != frozen_model["artifact_hashes"][name]:
            raise RuntimeError(f"frozen input artifact changed: {name}")
    probability = json.loads(
        (VALIDATION / "probability_package.json").read_text(encoding="utf-8")
    )
    benefit = json.loads((VALIDATION / "benefit_package.json").read_text(encoding="utf-8"))
    safety = json.loads((VALIDATION / "safety_package.json").read_text(encoding="utf-8"))
    selection = json.loads(
        (VALIDATION / "threshold_selection.json").read_text(encoding="utf-8")
    )
    bundle = V4PredictionBundle.from_packages(probability, benefit, safety)
    final_spec = split["final_holdout"]
    development_roles = json.loads(
        (MODEL_STUDY / "split_manifest.json").read_text(encoding="utf-8")
    )["roles"]
    development_ids = {
        case_id for case_ids in development_roles.values() for case_id in case_ids
    }
    rows = []
    for scenario in final_spec["scenarios"]:
        for seed in final_spec["seeds"]:
            for demand_scale in final_spec["demand_scale"]:
                for heterogeneity in final_spec["heterogeneity"]:
                    for penetration in final_spec["navigation_penetration"]:
                        case = build_alignment_case(
                            scenario=str(scenario),
                            seed=int(seed),
                            demand_scale=float(demand_scale),
                            heterogeneity=str(heterogeneity),
                            navigation_penetration=float(penetration),
                            user_count=6,
                            regret_limit=float(prereg["success_definition"]["regret_limit"]),
                            epsilon_grid=[0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.24],
                            minimum_relative_ttt_gain=float(
                                prereg["success_definition"]["minimum_relative_ttt_gain"]
                            ),
                            safety_delta=float(prereg["success_definition"]["safety_delta"]),
                            source_split="v4_final_holdout",
                        )
                        case["features_v3"] = {
                            name: float(case["features"][name]) for name in FEATURE_SCHEMA
                        }
                        case["features"] = expand_v4_features(case)
                        rows.append(case)
    if len(rows) != final_spec["expected_case_count"]:
        raise RuntimeError("v4 holdout case count differs from preregistration")
    holdout_ids = {row["case_id"] for row in rows}
    if holdout_ids & development_ids:
        raise RuntimeError("v4 final holdout case appeared in model fitting")
    matrix = np.asarray(
        [[row["features"][name] for name in V4_FEATURE_SCHEMA] for row in rows], dtype=float
    )
    prediction_started = time.perf_counter()
    prediction = bundle.predict(matrix)
    prediction_elapsed = time.perf_counter() - prediction_started
    decision_latency = prediction_elapsed / len(rows)
    v3_selected = json.loads((V3_STUDY / "selected_model.json").read_text(encoding="utf-8"))
    v3_frozen = yaml.safe_load((ROOT / "configs/v3/frozen_thresholds.yaml").read_text())
    v3_ensemble = BootstrapFeasibilityEnsemble.from_dict(v3_selected["ensemble"])
    v3_matrix = np.asarray(
        [[row["features_v3"][name] for name in FEATURE_SCHEMA] for row in rows], dtype=float
    )
    v3_probability, v3_uncertainty, v3_lower = v3_ensemble.predict(v3_matrix)
    v3_benefit = _regression_predict(v3_selected["benefit_regression"], v3_matrix)
    v3_benefit_lcb = v3_benefit - v3_frozen["benefit_lcb_z"] * v3_selected[
        "benefit_regression"
    ]["residual_standard_deviation"]
    v3_safety = _regression_predict(v3_selected["safety_regression"], v3_matrix)
    v3_safety_upper = v3_safety + v3_frozen["benefit_lcb_z"] * v3_selected[
        "safety_regression"
    ]["residual_standard_deviation"]
    v4_policies = {name: _v4_policy(selection, name) for name in ("V4-P", "V4-E", "V4-C")}
    final_name = frozen["selected_policy"]
    policies = {name: [] for name in ("B6", "V3-C", "V3-D", "V4-P", "V4-E", "V4-C", "V4-F")}
    decision_log = []
    for index, row in enumerate(rows):
        inputs = V4DecisionInputs(
            case_id=row["case_id"],
            success_probability=float(prediction["probability"][index]),
            success_probability_lower=float(prediction["probability_lower"][index]),
            expected_benefit=float(prediction["expected_benefit"][index]),
            benefit_lower=float(prediction["benefit_lower"][index]),
            safety_difference_upper=float(prediction["safety_difference_upper"][index]),
            safety_failure_probability=float(
                prediction["safety_failure_probability"][index]
            ),
            safety_failure_probability_upper=float(
                prediction["safety_failure_probability_upper"][index]
            ),
            esiv=float(prediction["esiv"][index]),
            esiv_lower=float(prediction["esiv_lower"][index]),
            legal=bool(row["adaptive_counterfactual"]["legal"]),
        )
        v4_decisions = {name: policy.decide(inputs) for name, policy in v4_policies.items()}
        v3_c = bool(
            v3_probability[index] >= v3_frozen["p_win_threshold"]
            and v3_uncertainty[index] <= v3_frozen["maximum_uncertainty"]
            and v3_safety_upper[index] <= v3_frozen["safety_delta"]
        )
        v3_d = bool(
            v3_c
            and v3_lower[index] > 0.5
            and row["features_v3"]["acceptance_probability"]
            >= v3_frozen["minimum_acceptance_probability"]
            and v3_benefit_lcb[index] >= v3_frozen["minimum_relative_ttt_gain"]
            and max(0.0, -v3_benefit_lcb[index]) <= v3_frozen["maximum_tail_loss"]
        )
        interventions = {
            "B6": True,
            "V3-C": v3_c,
            "V3-D": v3_d,
            "V4-P": v4_decisions["V4-P"].intervene,
            "V4-E": v4_decisions["V4-E"].intervene,
            "V4-C": v4_decisions["V4-C"].intervene,
            "V4-F": v4_decisions[final_name].intervene,
        }
        baseline_ttt = float(row["baseline_metrics"]["eta_only_ttt"])
        adaptive_ttt = float(row["adaptive_counterfactual"]["ttt"])
        success = row["label"] == "WIN"
        for name, intervene in interventions.items():
            selected_ttt = adaptive_ttt if intervene else baseline_ttt
            policies[name].append(
                {
                    "case_id": row["case_id"],
                    "intervene": intervene,
                    "success": bool(intervene and success),
                    "counterfactual_success": success,
                    "system_ttt_gain": baseline_ttt - selected_ttt,
                    "relative_ttt_gain": (baseline_ttt - selected_ttt) / max(baseline_ttt, 1e-9),
                    "baseline_ttt": baseline_ttt,
                    "selected_ttt": selected_ttt,
                    "maximum_regret": float(row["adaptive_counterfactual"]["maximum_regret"]),
                    "regret_violation": row["adaptive_counterfactual"]["maximum_regret"]
                    > frozen["regret_limit"] + 1e-10,
                    "safety_violation": row["adaptive_counterfactual"]["safety_difference"]
                    > frozen["safety_delta"] + 1e-10,
                    "legal_violation": not bool(row["adaptive_counterfactual"]["legal"]),
                    "decision_latency_seconds": decision_latency,
                }
            )
        final_decision = v4_decisions[final_name]
        decision_log.append(
            {
                **final_decision.__dict__,
                "outcome": "SUCCESS" if final_decision.intervene and success else "FAILURE" if final_decision.intervene else "ABSTAIN",
                "selected_v4_policy": final_name,
                "model_hash": model_hash_before,
                "threshold_hash": threshold_hash_before,
            }
        )
    metrics = {name: _extended_metrics(values) for name, values in policies.items()}
    final_metrics = metrics["V4-F"]
    final_groups = group_metrics(
        rows,
        [int(row["label"] == "WIN") for row in rows],
        [row["intervene"] for row in policies["V4-F"]],
    )
    constraints = (
        final_metrics["safety_violation_count"] == 0
        and final_metrics["regret_violation_count"] == 0
        and final_metrics["legal_violation_count"] == 0
        and final_metrics["mean_network_ttt_gain"] > 0
    )
    enough = final_metrics["intervention_count"] >= frozen["minimum_intervention_count"]
    if (
        final_metrics["intervention_precision"] >= 0.80
        and final_metrics["coverage"] >= 0.25
        and constraints
        and enough
    ):
        outcome = "S+"
    elif (
        final_metrics["intervention_precision"] >= 0.80
        and final_metrics["coverage"] >= 0.20
        and constraints
        and enough
    ):
        outcome = "S"
    elif final_metrics["intervention_precision"] >= 0.70 and constraints:
        outcome = "P"
    else:
        outcome = "F"
    v3_metrics = json.loads(
        (ROOT / "artifacts/studies/v3_selective_holdout/summary.json").read_text(
            encoding="utf-8"
        )
    )["primary_metrics"]
    statistical = {
        "H15_precision_at_least_0_80": final_metrics["intervention_precision"] >= 0.80,
        "H16_coverage_exceeds_v3D": final_metrics["coverage"] > v3_metrics["coverage"],
        "H16_coverage_at_least_0_20": final_metrics["coverage"] >= 0.20,
        "H17_zero_safety_violations": final_metrics["safety_violation_count"] == 0,
        "H18_PBR_exceeds_v3D": final_metrics["population_benefit_rate"]
        > v3_metrics["population_benefit_rate"],
        "strong_scientific_support": final_metrics["intervention_precision_ci95"][0] > 0.60,
        "very_strong_scientific_support": final_metrics["intervention_precision_ci95"][0] > 0.70,
        "worst_group_precision": final_groups["worst_group_precision"],
        "median_group_precision": final_groups["median_group_precision"],
        "minimum_intervention_count_met": enough,
    }
    if _sha(frozen_model_path) != model_hash_before or _sha(frozen_threshold_path) != threshold_hash_before:
        raise RuntimeError("v4 frozen configuration changed during holdout")
    summary = {
        "complete": True,
        "study": "Study XII — Frozen v4 Holdout",
        "untouched_holdout": True,
        "case_count": len(rows),
        "label_counts": dict(Counter(row["label"] for row in rows)),
        "selected_policy": final_name,
        "policy_metrics": metrics,
        "primary_metrics": final_metrics,
        "group_metrics": final_groups,
        "statistical_tests": statistical,
        "outcome": outcome,
        "outcome_text": {
            "S+": "High-precision CONCORDIA supported at stretch coverage.",
            "S": "High-precision CONCORDIA supported.",
            "P": "High-precision CONCORDIA partially supported.",
            "F": "High-precision CONCORDIA not supported.",
        }[outcome],
        "model_hash_before": model_hash_before,
        "model_hash_after": _sha(frozen_model_path),
        "threshold_hash_before": threshold_hash_before,
        "threshold_hash_after": _sha(frozen_threshold_path),
        "frozen_immutable": True,
        "holdout_case_ids_absent_from_development": not bool(holdout_ids & development_ids),
        "rl_used": False,
        "claim_boundary": prereg["claim_boundary"],
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    raw_path = OUTPUT / "raw_metrics.json"
    processed_path = OUTPUT / "processed_metrics.json"
    statistical_path = OUTPUT / "statistical_tests.json"
    summary_path = OUTPUT / "summary.json"
    decision_path = OUTPUT / "decision_log.json"
    policy_path = OUTPUT / "policy_rows.json"
    _write(raw_path, rows)
    _write(processed_path, metrics)
    _write(statistical_path, statistical)
    _write(summary_path, summary)
    _write(decision_path, decision_log)
    _write(policy_path, policies)
    figure_dir = OUTPUT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8.0, 4.5))
    names = list(metrics)
    axis.bar(names, [metrics[name]["intervention_precision"] for name in names], color="#444444")
    axis.plot(names, [metrics[name]["coverage"] for name in names], marker="o", color="#0b6e4f", label="coverage")
    axis.axhline(0.80, linestyle="--", color="#777777")
    axis.set_ylim(0, 1)
    axis.legend()
    fig.tight_layout()
    figure_path = figure_dir / "holdout_precision_coverage.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    ended = datetime.now(timezone.utc)
    outputs = (
        raw_path,
        processed_path,
        statistical_path,
        summary_path,
        decision_path,
        policy_path,
        figure_path,
    )
    registry = ExperimentRegistry(str(ROOT / "artifacts/runs")).create(
        {"seeds": final_spec["seeds"], "final_holdout": final_spec},
        summary,
        input_paths=(
            "configs/v4/frozen_model.yaml",
            "configs/v4/frozen_thresholds.yaml",
            "artifacts/v4/freeze_manifest.json",
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
            "model_hash": model_hash_before,
            "threshold_hash": threshold_hash_before,
            "feature_schema_hash": frozen_model["feature_schema_hash"],
            "data_split_hash": frozen_model["data_split_hash"],
            "result_hash": _sha(summary_path),
        },
    )
    manifest["final_holdout_started"] = True
    manifest["final_holdout_completed"] = True
    manifest["final_holdout_result_hash"] = _sha(summary_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(summary_path)
    return summary_path


if __name__ == "__main__":
    run()
