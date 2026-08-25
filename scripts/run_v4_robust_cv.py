#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from concordia.feasibility import (
    ProbabilityCalibrator,
    V4_FEATURE_SCHEMA,
    V4ProbabilityModel,
    build_v4_candidate_models,
    calibration_error,
    leave_group_out_folds,
    precision_constrained_threshold,
)


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "artifacts/studies/v4_model_selection"


def _write(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _fit_m5_with_internal_calibration(
    prototype: V4ProbabilityModel,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
) -> np.ndarray:
    split_at = max(4, int(len(train_x) * 0.80))
    fit_x, fit_y = train_x[:split_at], train_y[:split_at]
    cal_x, cal_y = train_x[split_at:], train_y[split_at:]
    model = prototype.clone(941).fit(fit_x, fit_y)
    raw_cal = model.predict_proba(cal_x)
    raw_test = model.predict_proba(test_x)
    if len(np.unique(cal_y)) < 2:
        return raw_test
    return ProbabilityCalibrator("isotonic").fit(raw_cal, cal_y).predict(raw_test)


def run() -> Path:
    config = yaml.safe_load(
        (ROOT / "configs/v4/model_selection.yaml").read_text(encoding="utf-8")
    )
    prereg = yaml.safe_load(
        (ROOT / "configs/v4/preregistration.yaml").read_text(encoding="utf-8")
    )
    rows = json.loads((STUDY / "raw_metrics.json").read_text(encoding="utf-8"))
    train = [row for row in rows if row["development_role"] == "training"]
    matrix = np.asarray(
        [[row["features"][name] for name in V4_FEATURE_SCHEMA] for row in train],
        dtype=float,
    )
    labels = np.asarray([row["targets"]["success"] for row in train], dtype=int)
    benefit = np.asarray([row["targets"]["relative_ttt_gain"] for row in train])
    folds = leave_group_out_folds(train)
    prototypes = build_v4_candidate_models(V4_FEATURE_SCHEMA, int(config["bootstrap_seed"]))
    results = {}
    for prototype in prototypes:
        fold_results = []
        all_labels = []
        all_probabilities = []
        for fold_index, fold in enumerate(folds):
            train_indices = np.asarray(fold["train_indices"], dtype=int)
            test_indices = np.asarray(fold["test_indices"], dtype=int)
            if len(np.unique(labels[train_indices])) < 2 or not len(test_indices):
                continue
            if prototype.name == "M5_calibrated_gradient_boosting":
                probabilities = _fit_m5_with_internal_calibration(
                    prototype,
                    matrix[train_indices],
                    labels[train_indices],
                    matrix[test_indices],
                )
            else:
                model = prototype.clone(1000 + fold_index * 20).fit(
                    matrix[train_indices], labels[train_indices]
                )
                probabilities = model.predict_proba(matrix[test_indices])
            operating = precision_constrained_threshold(
                labels[test_indices],
                probabilities,
                precision_target=float(prereg["primary"]["intervention_precision_target"]),
                coverage_guard=float(prereg["primary"]["coverage_guard"]),
                thresholds=config["probability_thresholds"],
            )
            selected = probabilities >= operating["selected"]["threshold"]
            mean_gain = float(benefit[test_indices][selected].mean()) if selected.any() else 0.0
            fold_results.append(
                {
                    "family": fold["family"],
                    "group": fold["group"],
                    **operating["selected"],
                    "threshold_feasible": operating["feasible"],
                    "mean_relative_ttt_gain": mean_gain,
                }
            )
            all_labels.extend(labels[test_indices].tolist())
            all_probabilities.extend(probabilities.tolist())
        precisions = [row["precision"] for row in fold_results]
        coverages = [row["coverage"] for row in fold_results]
        gains = [row["mean_relative_ttt_gain"] for row in fold_results]
        family_summary = defaultdict(list)
        for row in fold_results:
            family_summary[row["family"]].append(row["precision"])
        calibration = calibration_error(all_labels, all_probabilities)
        results[prototype.name] = {
            "folds": fold_results,
            "fold_count": len(fold_results),
            "worst_group_precision": float(min(precisions)),
            "p10_group_precision": float(np.percentile(precisions, 10)),
            "median_group_precision": float(np.median(precisions)),
            "mean_coverage": float(np.mean(coverages)),
            "mean_selected_gain": float(np.mean(gains)),
            "calibration": calibration,
            "family_worst_precision": {
                family: float(min(values)) for family, values in family_summary.items()
            },
        }
    target = float(prereg["primary"]["intervention_precision_target"])
    selected_name = max(
        results,
        key=lambda name: (
            results[name]["worst_group_precision"] >= target,
            results[name]["p10_group_precision"] >= target,
            results[name]["p10_group_precision"],
            results[name]["mean_coverage"],
            -results[name]["calibration"]["ece"],
            results[name]["mean_selected_gain"],
        ),
    )
    trained = json.loads((STUDY / "trained_candidates.json").read_text(encoding="utf-8"))
    selected_model = next(
        model for model in trained["candidate_models"] if model["name"] == selected_name
    )
    justification = {
        "objective": [
            "worst-group precision >= 0.80",
            "p10 group precision >= 0.80",
            "maximize coverage",
            "minimize ECE",
            "maximize selected mean TTT gain",
        ],
        "selected_model": selected_name,
        "selected_metrics": results[selected_name],
        "target_met_in_robust_cv": results[selected_name]["worst_group_precision"] >= target,
        "final_holdout_used": False,
    }
    _write(STUDY / "robust_cv_results.json", results)
    _write(STUDY / "selected_base_model.json", selected_model)
    _write(STUDY / "selected_model_justification.json", justification)
    _write(
        STUDY / "summary.json",
        {
            "complete": True,
            "study": "Study X — Robust Feasibility Model Selection",
            "selected_model": selected_name,
            "worst_group_precision": results[selected_name]["worst_group_precision"],
            "p10_group_precision": results[selected_name]["p10_group_precision"],
            "mean_coverage": results[selected_name]["mean_coverage"],
            "ece": results[selected_name]["calibration"]["ece"],
            "final_holdout_used": False,
        },
    )
    print(STUDY / "summary.json")
    return STUDY / "summary.json"


if __name__ == "__main__":
    run()
