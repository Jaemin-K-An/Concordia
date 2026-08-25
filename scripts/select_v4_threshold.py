#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/concordia-matplotlib-v4")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/concordia-cache-v4")

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from concordia.feasibility import (
    FeasibilityModel,
    V4PredictionBundle,
    V4_FEATURE_SCHEMA,
    calibration_error,
    false_safe_rate,
    group_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_STUDY = ROOT / "artifacts/studies/v4_model_selection"
STUDY = ROOT / "artifacts/studies/v4_precision_validation"


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _matrix(rows: list[dict]) -> np.ndarray:
    return np.asarray(
        [[row["features"][name] for name in V4_FEATURE_SCHEMA] for row in rows],
        dtype=float,
    )


def _group_quantiles(group_result: dict) -> tuple[float, float, float]:
    values = [
        group["precision"]
        for dimension in group_result["dimensions"].values()
        for group in dimension.values()
        if group["precision"] is not None
    ]
    return (
        float(min(values)) if values else 0.0,
        float(np.percentile(values, 10)) if values else 0.0,
        float(np.median(values)) if values else 0.0,
    )


def _operating_candidate(
    rows: list[dict],
    labels: np.ndarray,
    benefit: np.ndarray,
    selected: np.ndarray,
    *,
    policy: str,
    score_threshold: float,
    benefit_threshold: float,
) -> dict:
    count = int(selected.sum())
    successes = int(labels[selected].sum())
    precision = successes / max(1, count)
    coverage = count / max(1, len(labels))
    groups = group_metrics(rows, labels, selected)
    worst, p10, median = _group_quantiles(groups)
    return {
        "policy": policy,
        "score_threshold": float(score_threshold),
        "benefit_threshold": float(benefit_threshold),
        "intervention_count": count,
        "successful_intervention_count": successes,
        "precision": precision,
        "coverage": coverage,
        "population_benefit_rate": successes / max(1, len(labels)),
        "mean_relative_ttt_gain": float(benefit[selected].mean()) if count else 0.0,
        "worst_group_precision": worst,
        "p10_group_precision": p10,
        "median_group_precision": median,
        "group_metrics": groups,
    }


def _select_operating(candidates: list[dict], prereg: dict) -> dict:
    precision_target = float(prereg["primary"]["intervention_precision_target"])
    guard = float(prereg["primary"]["coverage_guard"])
    feasible = [
        row for row in candidates if row["precision"] >= precision_target and row["coverage"] >= guard
    ]
    if feasible:
        selected = max(
            feasible,
            key=lambda row: (
                row["worst_group_precision"] >= 0.60,
                row["median_group_precision"] >= 0.80,
                row["p10_group_precision"],
                row["coverage"],
                row["mean_relative_ttt_gain"],
            ),
        )
    else:
        guarded = [row for row in candidates if row["coverage"] >= guard]
        selected = max(
            guarded or candidates,
            key=lambda row: (
                row["precision"],
                row["p10_group_precision"],
                row["coverage"],
                row["mean_relative_ttt_gain"],
            ),
        )
    return {**selected, "primary_validation_constraint_met": bool(feasible)}


def _interaction_analysis(training: list[dict], validation: list[dict]) -> dict:
    training_x = _matrix(training)
    training_y = np.asarray([row["targets"]["success"] for row in training])
    validation_x = _matrix(validation)
    validation_y = np.asarray([row["targets"]["success"] for row in validation])
    interaction_index = V4_FEATURE_SCHEMA.index("heterogeneity_rad_interaction")
    reduced_indices = tuple(
        index for index in range(len(V4_FEATURE_SCHEMA)) if index != interaction_index
    )
    full = FeasibilityModel(
        "H19_full", "logistic", V4_FEATURE_SCHEMA, regularization=0.03, iterations=1800
    ).fit(training_x, training_y)
    reduced = FeasibilityModel(
        "H19_reduced",
        "logistic",
        V4_FEATURE_SCHEMA,
        feature_indices=reduced_indices,
        regularization=0.03,
        iterations=1800,
    ).fit(training_x, training_y)
    full_probability = np.clip(full.predict_proba(validation_x), 1e-8, 1.0 - 1e-8)
    reduced_probability = np.clip(reduced.predict_proba(validation_x), 1e-8, 1.0 - 1e-8)

    def log_loss(probability):
        return float(
            -np.mean(
                validation_y * np.log(probability)
                + (1 - validation_y) * np.log(1 - probability)
            )
        )

    mean = np.asarray(full.parameters["mean"])
    scale = np.asarray(full.parameters["scale"])
    normalized = (training_x - mean) / scale
    probability = full.predict_proba(training_x)
    design = np.column_stack([np.ones(len(normalized)), normalized])
    weights = probability * (1.0 - probability)
    hessian = design.T @ (design * weights[:, None]) + np.eye(design.shape[1]) * 0.03
    covariance = np.linalg.pinv(hessian)
    coefficient = float(full.parameters["weights"][interaction_index])
    standard_error = float(np.sqrt(max(covariance[interaction_index + 1, interaction_index + 1], 0.0)))
    return {
        "interaction": "heterogeneity_rad_interaction",
        "standardized_coefficient": coefficient,
        "approximate_ci95": [
            coefficient - 1.96 * standard_error,
            coefficient + 1.96 * standard_error,
        ],
        "validation_log_loss_full": log_loss(full_probability),
        "validation_log_loss_reduced": log_loss(reduced_probability),
        "incremental_information": log_loss(reduced_probability) - log_loss(full_probability),
        "supports_H19": log_loss(full_probability) < log_loss(reduced_probability),
    }


def run() -> Path:
    config = yaml.safe_load(
        (ROOT / "configs/v4/model_selection.yaml").read_text(encoding="utf-8")
    )
    prereg = yaml.safe_load(
        (ROOT / "configs/v4/preregistration.yaml").read_text(encoding="utf-8")
    )
    rows = json.loads((MODEL_STUDY / "raw_metrics.json").read_text(encoding="utf-8"))
    validation = [row for row in rows if row["development_role"] == "validation"]
    development = [row for row in rows if row["development_role"] != "validation"]
    probability_package = json.loads(
        (STUDY / "probability_package.json").read_text(encoding="utf-8")
    )
    benefit_package = json.loads(
        (STUDY / "benefit_package.json").read_text(encoding="utf-8")
    )
    safety_package = json.loads(
        (STUDY / "safety_package.json").read_text(encoding="utf-8")
    )
    bundle = V4PredictionBundle.from_packages(
        probability_package, benefit_package, safety_package
    )
    matrix = _matrix(validation)
    started = time.perf_counter()
    prediction = bundle.predict(matrix)
    batch_latency = time.perf_counter() - started
    case_latencies = []
    for index in range(len(matrix)):
        start = time.perf_counter()
        bundle.predict(matrix[index : index + 1])
        case_latencies.append(time.perf_counter() - start)
    labels = np.asarray([row["targets"]["success"] for row in validation], dtype=int)
    realized_benefit = np.asarray(
        [row["targets"]["relative_ttt_gain"] for row in validation], dtype=float
    )
    safety_eligible = (
        prediction["safety_difference_upper"]
        <= float(prereg["success_definition"]["safety_delta"])
    ) & (
        prediction["safety_failure_probability_upper"]
        <= float(config["safety_failure_probability_threshold"])
    )
    candidates = {"V4-P": [], "V4-E": [], "V4-C": []}
    for benefit_threshold in config["benefit_thresholds"]:
        benefit_eligible = prediction["benefit_lower"] >= float(benefit_threshold)
        for threshold in config["probability_thresholds"]:
            selected = (
                prediction["probability"] >= float(threshold)
            ) & benefit_eligible & safety_eligible
            candidates["V4-P"].append(
                _operating_candidate(
                    validation,
                    labels,
                    realized_benefit,
                    selected,
                    policy="V4-P",
                    score_threshold=float(threshold),
                    benefit_threshold=float(benefit_threshold),
                )
            )
        conformal_threshold = float(
            probability_package["conformal_controller"]["probability_threshold"]
        )
        selected = (
            prediction["probability"] >= conformal_threshold
        ) & benefit_eligible & safety_eligible
        candidates["V4-C"].append(
            _operating_candidate(
                validation,
                labels,
                realized_benefit,
                selected,
                policy="V4-C",
                score_threshold=conformal_threshold,
                benefit_threshold=float(benefit_threshold),
            )
        )
    for threshold in config["esiv_thresholds"]:
        selected = (prediction["esiv_lower"] >= float(threshold)) & safety_eligible
        candidates["V4-E"].append(
            _operating_candidate(
                validation,
                labels,
                realized_benefit,
                selected,
                policy="V4-E",
                score_threshold=float(threshold),
                benefit_threshold=0.0,
            )
        )
    selected_policies = {
        policy: _select_operating(values, prereg) for policy, values in candidates.items()
    }
    final = _select_operating(list(selected_policies.values()), prereg)
    final_policy = final["policy"]
    actual_safety = [row["targets"]["safety_difference"] for row in validation]
    safety_validation = false_safe_rate(
        actual_safety,
        prediction["safety_difference_upper"],
        float(prereg["success_definition"]["safety_delta"]),
    )
    calibration = calibration_error(labels, prediction["probability"])
    interaction = _interaction_analysis(development, validation)
    selected_package = {
        "complete": True,
        "selected_policy": final_policy,
        "selected_operating_point": final,
        "policy_operating_points": selected_policies,
        "probability_threshold": final["score_threshold"] if final_policy != "V4-E" else 0.0,
        "esiv_threshold": final["score_threshold"] if final_policy == "V4-E" else 0.0,
        "benefit_threshold": final["benefit_threshold"],
        "safety_delta": float(prereg["success_definition"]["safety_delta"]),
        "safety_failure_probability_threshold": float(
            config["safety_failure_probability_threshold"]
        ),
        "validation_case_ids": [row["case_id"] for row in validation],
        "final_holdout_case_ids": [],
        "final_holdout_used": False,
    }
    output = STUDY / "threshold_selection.json"
    _write(output, selected_package)
    _write(STUDY / "precision_coverage_curves.json", candidates)
    _write(
        STUDY / "validation_predictions.json",
        [
            {
                "case_id": row["case_id"],
                "label": int(labels[index]),
                **{name: float(values[index]) for name, values in prediction.items()},
            }
            for index, row in enumerate(validation)
        ],
    )
    esiv_coverage_at_precision80 = (
        selected_policies["V4-E"]["coverage"]
        if selected_policies["V4-E"]["precision"] >= 0.80
        else 0.0
    )
    probability_coverage_at_precision80 = (
        selected_policies["V4-P"]["coverage"]
        if selected_policies["V4-P"]["precision"] >= 0.80
        else 0.0
    )
    statistics = {
        "calibration": calibration,
        "safety_false_safe": safety_validation,
        "H19_interaction": interaction,
        "H20_ESIV_coverage_at_precision80": esiv_coverage_at_precision80,
        "H20_probability_coverage_at_precision80": probability_coverage_at_precision80,
        "H20_ESIV_better": esiv_coverage_at_precision80
        > probability_coverage_at_precision80,
        "latency": {
            "batch_seconds": batch_latency,
            "mean_per_case_seconds": float(np.mean(case_latencies)),
            "p95_per_case_seconds": float(np.percentile(case_latencies, 95)),
        },
    }
    _write(STUDY / "statistical_tests.json", statistics)
    summary = {
        "complete": True,
        "study": "Study XI — Precision-Constrained Validation",
        "selected_policy": final_policy,
        "selected_operating_point": final,
        "policy_operating_points": selected_policies,
        "validation_ece": calibration["ece"],
        "validation_ece_target_met": calibration["ece"] < 0.05,
        "safety_false_safe_count": safety_validation["false_safe_count"],
        "final_holdout_used": False,
        "claim_boundary": "Development validation only; not primary v4 success evidence.",
    }
    _write(STUDY / "summary.json", summary)
    figure_dir = STUDY / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(6.5, 4.4))
    for policy, values in candidates.items():
        unique = sorted(
            {(row["coverage"], row["precision"]) for row in values}, key=lambda item: item[0]
        )
        axis.plot(
            [row[0] for row in unique],
            [row[1] for row in unique],
            marker=".",
            label=policy,
        )
    axis.axhline(0.80, linestyle="--", color="#777777")
    axis.axvline(0.20, linestyle=":", color="#777777")
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Coverage", ylabel="Precision")
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "precision_at_coverage.png", dpi=180)
    plt.close(fig)
    print(output)
    return output


if __name__ == "__main__":
    run()
