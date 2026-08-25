#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import Counter
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


ROOT = Path(__file__).resolve().parents[1]
MODEL_STUDY = ROOT / "artifacts" / "studies" / "v3_feasibility_prediction"
OUTPUT = ROOT / "artifacts" / "studies" / "v3_selective_holdout"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _regression_predict(package: dict, matrix: np.ndarray) -> np.ndarray:
    mean = np.asarray(package["mean"])
    scale = np.asarray(package["scale"])
    weights = np.asarray(package["coefficients"])
    return package["intercept"] + ((matrix - mean) / scale) @ weights


def run() -> Path:
    source_commit, source_dirty = capture_source_state()
    started = datetime.now(timezone.utc)
    split = yaml.safe_load((ROOT / "configs/v3/splits.yaml").read_text())
    prereg = yaml.safe_load((ROOT / "configs/v3/preregistration.yaml").read_text())
    model_config = yaml.safe_load((ROOT / "configs/v3/model_selection.yaml").read_text())
    freeze_path = ROOT / model_config["freeze_file"]
    freeze_manifest_path = ROOT / model_config["freeze_manifest"]
    if not freeze_path.is_file() or not freeze_manifest_path.is_file():
        raise RuntimeError("holdout is forbidden before threshold freeze")
    freeze_hash_before = _sha(freeze_path)
    freeze_manifest = json.loads(freeze_manifest_path.read_text(encoding="utf-8"))
    if freeze_hash_before != freeze_manifest["threshold_config_hash"]:
        raise RuntimeError("threshold checksum differs from freeze manifest")
    frozen = yaml.safe_load(freeze_path.read_text(encoding="utf-8"))
    selected = json.loads((MODEL_STUDY / "selected_model.json").read_text(encoding="utf-8"))
    if _sha(MODEL_STUDY / "selected_model.json") != frozen["selected_model_hash"]:
        raise RuntimeError("selected model changed after freeze")
    train_ids = set(selected["training_case_ids"])
    validation_ids = set(selected["validation_case_ids"])
    holdout_spec = split["holdout"]
    if set(holdout_spec["seeds"]) & set(split["training"]["seeds"] + split["validation"]["seeds"]):
        raise RuntimeError("holdout seed leakage")
    rows = []
    for scenario in holdout_spec["scenarios"]:
        for seed in holdout_spec["seeds"]:
            for demand_scale in holdout_spec["demand_scale"]:
                for heterogeneity in holdout_spec["heterogeneity"]:
                    for penetration in holdout_spec["navigation_penetration"]:
                        rows.append(
                            build_alignment_case(
                                scenario=str(scenario),
                                seed=int(seed),
                                demand_scale=float(demand_scale),
                                heterogeneity=str(heterogeneity),
                                navigation_penetration=float(penetration),
                                user_count=6,
                                regret_limit=float(prereg["success_definition"]["regret_limit"]),
                                epsilon_grid=[0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.24],
                                minimum_relative_ttt_gain=float(frozen["minimum_relative_ttt_gain"]),
                                safety_delta=float(frozen["safety_delta"]),
                                source_split="holdout",
                            )
                        )
    holdout_ids = {row["case_id"] for row in rows}
    if holdout_ids & (train_ids | validation_ids):
        raise RuntimeError("holdout case ID appeared in model fitting")
    matrix = np.asarray(
        [[row["features"][name] for name in FEATURE_SCHEMA] for row in rows], dtype=float
    )
    ensemble = BootstrapFeasibilityEnsemble.from_dict(selected["ensemble"])
    probabilities, uncertainty, lower = ensemble.predict(matrix)
    benefit_prediction = _regression_predict(selected["benefit_regression"], matrix)
    benefit_lcb = benefit_prediction - float(frozen["benefit_lcb_z"]) * float(
        selected["benefit_regression"]["residual_standard_deviation"]
    )
    safety_prediction = _regression_predict(selected["safety_regression"], matrix)
    safety_upper = safety_prediction + float(frozen["benefit_lcb_z"]) * float(
        selected["safety_regression"]["residual_standard_deviation"]
    )
    full_policy = SelectiveInterventionPolicy(
        FeasibilityGate(
            probability_threshold=float(frozen["p_win_threshold"]),
            maximum_uncertainty=float(frozen["maximum_uncertainty"]),
            safety_delta=float(frozen["safety_delta"]),
            minimum_acceptance_probability=float(frozen["minimum_acceptance_probability"]),
            minimum_ttt_lcb_gain=float(frozen["minimum_relative_ttt_gain"]),
            maximum_tail_loss=float(frozen["maximum_tail_loss"]),
        )
    )
    policy_rows: dict[str, list[dict]] = {
        name: [] for name in ("B6", "V3-A", "V3-B", "V3-C", "V3-D")
    }
    decision_log = []
    for index, row in enumerate(rows):
        features = row["features"]
        counterfactual = row["adaptive_counterfactual"]
        win = row["label"] == "WIN"
        full = full_policy.decide(
            case_id=row["case_id"],
            p_win=float(probabilities[index]),
            p_win_lower=float(lower[index]),
            uncertainty=float(uncertainty[index]),
            alignment_potential=float(features["alignment_potential_score"]),
            route_overlap=float(features["route_overlap"]),
            safety_upper_difference=float(safety_upper[index]),
            acceptance_probability=float(features["acceptance_probability"]),
            ttt_lcb_gain=float(benefit_lcb[index]),
            predicted_tail_loss=max(0.0, -float(benefit_lcb[index])),
            legal=bool(row["baseline_metrics"]["legal"]),
            counterfactual_success=win,
        )
        decisions = {
            "B6": True,
            "V3-A": features["alignment_potential_score"] >= frozen["aps_threshold"],
            "V3-B": probabilities[index] >= frozen["p_win_threshold"]
            and uncertainty[index] <= frozen["maximum_uncertainty"],
            "V3-C": probabilities[index] >= frozen["p_win_threshold"]
            and uncertainty[index] <= frozen["maximum_uncertainty"]
            and safety_upper[index] <= frozen["safety_delta"],
            "V3-D": full.intervene,
        }
        for policy_name, intervene in decisions.items():
            baseline_ttt = float(row["baseline_metrics"]["eta_only_ttt"])
            adaptive_ttt = float(counterfactual["ttt"])
            selected_ttt = adaptive_ttt if intervene else baseline_ttt
            policy_rows[policy_name].append(
                {
                    "case_id": row["case_id"],
                    "intervene": bool(intervene),
                    "success": bool(intervene and win),
                    "counterfactual_success": win,
                    "system_ttt_gain": baseline_ttt - selected_ttt,
                    "baseline_ttt": baseline_ttt,
                    "selected_ttt": selected_ttt,
                    "regret_violation": counterfactual["maximum_regret"]
                    > prereg["success_definition"]["regret_limit"] + 1e-10,
                    "safety_violation": counterfactual["safety_difference"]
                    > frozen["safety_delta"] + 1e-10,
                    "legal_violation": not bool(counterfactual["legal"]),
                }
            )
        decision_log.append(
            {
                **full.__dict__,
                "outcome": full.outcome.value,
                "features": features,
                "threshold_config_hash": freeze_hash_before,
                "policy": "concordia-v3",
            }
        )
    metrics = {
        policy: summarize_selective_policy(values) for policy, values in policy_rows.items()
    }

    def ablation_rows(rule) -> list[dict]:
        values = []
        for index, b6_row in enumerate(policy_rows["B6"]):
            intervene = bool(rule(rows[index]))
            baseline_ttt = float(b6_row["baseline_ttt"])
            adaptive_ttt = float(b6_row["selected_ttt"])
            values.append(
                {
                    **b6_row,
                    "intervene": intervene,
                    "success": bool(intervene and b6_row["counterfactual_success"]),
                    "selected_ttt": adaptive_ttt if intervene else baseline_ttt,
                    "system_ttt_gain": baseline_ttt - adaptive_ttt if intervene else 0.0,
                }
            )
        return values

    ablations = {
        "A0_no_gate": metrics["B6"],
        "A1_APS_only": metrics["V3-A"],
        "A2_topology_only": summarize_selective_policy(
            ablation_rows(
                lambda row: row["features"]["route_overlap"]
                <= frozen["maximum_route_overlap"]
            )
        ),
        "A3_preference_only": summarize_selective_policy(
            ablation_rows(
                lambda row: row["features"]["preference_variance"]
                >= frozen["minimum_preference_variance"]
            )
        ),
        "A4_APS_topology": summarize_selective_policy(
            ablation_rows(
                lambda row: row["features"]["alignment_potential_score"]
                >= frozen["aps_threshold"]
                and row["features"]["route_overlap"]
                <= frozen["maximum_route_overlap"]
            )
        ),
        "A5_learned": metrics["V3-B"],
        "A6_learned_safety": metrics["V3-C"],
        "A7_full_v3": metrics["V3-D"],
    }
    primary = metrics["V3-D"]
    safety_pass = primary["safety_violation_count"] == 0
    if (
        primary["intervention_precision"] >= frozen["engineering_precision_target"]
        and primary["coverage"] >= frozen["coverage_target"]
        and primary["mean_network_ttt_gain"] > 0
        and safety_pass
    ):
        outcome = "S"
    elif (
        primary["intervention_precision"] > 0.50
        and primary["mean_network_ttt_gain"] > 0
        and safety_pass
    ):
        outcome = "P"
    else:
        outcome = "F"
    b6_failure_rate = 1.0 - metrics["B6"]["intervention_precision"]
    v3_failure_rate = 1.0 - primary["intervention_precision"] if primary["intervention_count"] else 1.0
    statistical = {
        "paired_ttt_B1_vs_V3D": paired_comparison(
            [item["baseline_ttt"] for item in policy_rows["V3-D"]],
            [item["selected_ttt"] for item in policy_rows["V3-D"]],
            seed=107,
        ),
        "H8_scientific_lower_ci_above_half": primary["intervention_precision_ci95"][0] > 0.50,
        "H8_point_precision_above_half": primary["intervention_precision"] > 0.50,
        "H9_failure_rate_reduced_vs_B6": v3_failure_rate < b6_failure_rate,
        "H10_mean_network_cost_noninferior": primary["mean_network_ttt_gain"] >= 0.0,
        "H13_overlap_win_correlation": float(
            np.corrcoef(
                [row["features"]["route_overlap"] for row in rows],
                [int(row["label"] == "WIN") for row in rows],
            )[0, 1]
        ),
        "H14_interaction_win_correlation": float(
            np.corrcoef(
                [row["features"]["heterogeneity_rad_interaction"] for row in rows],
                [int(row["label"] == "WIN") for row in rows],
            )[0, 1]
        ),
    }
    freeze_hash_after = _sha(freeze_path)
    if freeze_hash_after != freeze_hash_before:
        raise RuntimeError("threshold checksum changed during holdout")
    summary = {
        "complete": True,
        "study": "Study VI — Selective Intervention Holdout",
        "untouched_holdout": True,
        "case_count": len(rows),
        "label_counts": dict(Counter(row["label"] for row in rows)),
        "policy_metrics": metrics,
        "primary_policy": "V3-D",
        "primary_metrics": primary,
        "outcome": outcome,
        "outcome_text": {
            "S": "Selective CONCORDIA supported.",
            "P": "Selective CONCORDIA partially supported.",
            "F": "Selective CONCORDIA not supported.",
        }[outcome],
        "always_on_conclusion": "Always-on Adaptive Navigation: rejected as universal policy.",
        "threshold_hash_before": freeze_hash_before,
        "threshold_hash_after": freeze_hash_after,
        "threshold_immutable": True,
        "holdout_case_ids_absent_from_training": not bool(holdout_ids & train_ids),
        "statistical_tests": statistical,
        "claim_boundary": prereg["claim_boundary"],
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    raw_path = OUTPUT / "raw_metrics.json"
    processed_path = OUTPUT / "processed_metrics.json"
    statistical_path = OUTPUT / "statistical_tests.json"
    summary_path = OUTPUT / "summary.json"
    decisions_path = OUTPUT / "decision_log.json"
    ablation_path = OUTPUT / "ablations.json"
    _write(raw_path, rows)
    _write(processed_path, metrics)
    _write(statistical_path, statistical)
    _write(summary_path, summary)
    _write(decisions_path, decision_log)
    _write(ablation_path, ablations)
    figure_dir = OUTPUT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7.0, 4.5))
    names = list(metrics)
    axis.bar(names, [metrics[name]["intervention_precision"] for name in names], color="#444444")
    axis.plot(names, [metrics[name]["coverage"] for name in names], marker="o", color="#111111", label="coverage")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Rate")
    axis.legend()
    fig.tight_layout()
    policy_figure = figure_dir / "policy_precision_coverage.png"
    fig.savefig(policy_figure, dpi=180)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(6.2, 4.2))
    thresholds = np.linspace(0.30, 0.90, 25)
    curve = []
    labels = np.asarray([int(row["label"] == "WIN") for row in rows])
    for threshold in thresholds:
        selected_mask = probabilities >= threshold
        curve.append(
            (
                float(selected_mask.mean()),
                float(labels[selected_mask].mean()) if selected_mask.any() else 0.0,
            )
        )
    axis.plot([item[0] for item in curve], [item[1] for item in curve], marker=".")
    axis.scatter([primary["coverage"]], [primary["intervention_precision"]], color="red", label="frozen V3-D")
    axis.set(xlabel="Coverage", ylabel="Precision", title="Holdout intervention risk–coverage")
    axis.legend()
    fig.tight_layout()
    risk_figure = figure_dir / "holdout_risk_coverage.png"
    fig.savefig(risk_figure, dpi=180)
    plt.close(fig)
    ended = datetime.now(timezone.utc)
    registry_config = {
        "seeds": holdout_spec["seeds"],
        "holdout": holdout_spec,
        "frozen_threshold_hash": freeze_hash_before,
    }
    outputs = (
        raw_path,
        processed_path,
        statistical_path,
        summary_path,
        decisions_path,
        ablation_path,
        policy_figure,
        risk_figure,
    )
    run_dir = ExperimentRegistry(str(ROOT / "artifacts/runs")).create(
        registry_config,
        summary,
        input_paths=(
            "configs/v3/preregistration.yaml",
            "configs/v3/splits.yaml",
            "configs/v3/frozen_thresholds.yaml",
            "artifacts/studies/v3_feasibility_prediction/selected_model.json",
        ),
        external_output_paths=tuple(str(path.relative_to(ROOT)) for path in outputs),
        started_at=started,
        ended_at=ended,
        source_commit=source_commit,
        source_dirty=source_dirty,
    )
    shutil.copyfile(run_dir / "manifest.json", OUTPUT / "manifest.json")
    explicit_registry = {
        "git_commit": source_commit,
        "git_dirty": source_dirty,
        "model_hash": _sha(MODEL_STUDY / "selected_model.json"),
        "feature_schema_hash": frozen["feature_schema_hash"],
        "threshold_config_hash": freeze_hash_before,
        "data_split_hash": frozen["data_split_hash"],
        "result_hash": _sha(summary_path),
        "threshold_immutable": True,
    }
    _write(OUTPUT / "v3_registry.json", explicit_registry)
    freeze_manifest["holdout_started"] = True
    freeze_manifest["holdout_completed"] = True
    freeze_manifest["holdout_result_hash"] = _sha(summary_path)
    freeze_manifest_path.write_text(
        json.dumps(freeze_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(summary_path)
    return summary_path


if __name__ == "__main__":
    run()
