#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from concordia.feasibility import (
    RegimeProbabilityCalibrator,
    V5SuccessModel,
    V5_FEATURE_SCHEMA,
    unified_calibration_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_STUDY = ROOT / "artifacts/studies/v5_model_selection"
STUDY = ROOT / "artifacts/studies/v5_shift_detection"


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _matrix(rows: list[dict]) -> np.ndarray:
    return np.asarray(
        [[row["features"][name] for name in V5_FEATURE_SCHEMA] for row in rows], dtype=float
    )


def _frontier(labels: np.ndarray, probability: np.ndarray, rows: list[dict], thresholds) -> dict:
    values = []
    for threshold in thresholds:
        selected = probability >= float(threshold)
        count = int(selected.sum())
        precision = float(labels[selected].mean()) if count else 0.0
        coverage = count / max(1, len(labels))
        critical = []
        for dimension in ("navigation_penetration", "scenario"):
            groups = {}
            for index, row in enumerate(rows):
                key = (
                    str(row["condition"][dimension])
                    if dimension == "navigation_penetration"
                    else str(row["scenario"])
                )
                groups.setdefault(key, []).append(index)
            for indices in groups.values():
                mask = np.zeros(len(rows), dtype=bool)
                mask[indices] = True
                activated = mask & selected
                if int(activated.sum()) >= 10:
                    critical.append(float(labels[activated].mean()))
        values.append(
            {
                "threshold": float(threshold),
                "intervention_count": count,
                "precision": precision,
                "coverage": coverage,
                "critical_group_precision": min(critical) if critical else None,
                "critical_group_evidence_count": len(critical),
            }
        )
    feasible = [
        row
        for row in values
        if row["precision"] >= 0.80
        and (
            row["critical_group_precision"] is None
            or row["critical_group_precision"] >= 0.70
        )
    ]
    selected = max(
        feasible or values,
        key=lambda row: (
            row["precision"] >= 0.80,
            row["critical_group_precision"] is None
            or row["critical_group_precision"] >= 0.70,
            row["coverage"] if row["precision"] >= 0.80 else row["precision"],
            row["precision"],
        ),
    )
    return {"selected": selected, "curve": values, "feasible": bool(feasible)}


def run() -> Path:
    existing = STUDY / "calibrated_model.json"
    if (ROOT / "configs/v5/frozen_model.yaml").is_file():
        if not existing.is_file():
            raise RuntimeError("v5 is frozen but calibrated package is missing")
        print(existing)
        return existing
    config = yaml.safe_load((ROOT / "configs/v5/model.yaml").read_text())
    rows = json.loads((MODEL_STUDY / "enriched_rows.json").read_text())
    fit = [row for row in rows if row["development_role"] == "calibration_fit"]
    evaluation = [row for row in rows if row["development_role"] == "validation"]
    trained = json.loads((MODEL_STUDY / "trained_candidates.json").read_text())
    fit_y = np.asarray([row["targets"]["success"] for row in fit], dtype=int)
    evaluation_y = np.asarray(
        [row["targets"]["success"] for row in evaluation], dtype=int
    )
    results = []
    packages = {}
    for model_package in trained["candidate_models"]:
        model = V5SuccessModel.from_dict(model_package)
        raw_fit = model.predict_proba(_matrix(fit), [row["regime"] for row in fit])
        raw_evaluation = model.predict_proba(
            _matrix(evaluation), [row["regime"] for row in evaluation]
        )
        for method in config["calibration_methods"]:
            calibrator = RegimeProbabilityCalibrator.fit(
                method,
                raw_fit,
                fit_y,
                [row["regime"] for row in fit],
                minimum_regime_size=30,
            )
            probability = calibrator.predict(
                raw_evaluation, [row["regime"] for row in evaluation]
            )
            metrics = unified_calibration_metrics(evaluation_y, probability)
            frontier = _frontier(
                evaluation_y, probability, evaluation, config["probability_thresholds"]
            )
            regime_metrics = {}
            for regime in sorted({row["regime"] for row in evaluation}):
                mask = np.asarray([row["regime"] == regime for row in evaluation])
                regime_metrics[regime] = unified_calibration_metrics(
                    evaluation_y[mask], probability[mask]
                )
            key = f"{model.name}::{method}"
            result = {
                "key": key,
                "model": model.name,
                "calibration": method,
                "overall": metrics,
                "by_regime": regime_metrics,
                "frontier": frontier,
            }
            results.append(result)
            packages[key] = {
                "model": model.to_dict(),
                "calibrator": calibrator.to_dict(),
            }
    selected = max(
        results,
        key=lambda row: (
            row["frontier"]["feasible"],
            row["frontier"]["selected"]["coverage"],
            -row["overall"]["ece"],
            -row["overall"]["brier_score"],
        ),
    )
    package = {
        "complete": True,
        "selected_key": selected["key"],
        "selected_model": selected["model"],
        "selected_calibration": selected["calibration"],
        **packages[selected["key"]],
        "benefit_model": trained["benefit_model"],
        "calibration_protocol": selected["overall"]["protocol"],
        "calibration_fit_case_ids": [row["case_id"] for row in fit],
        "evaluation_case_ids": [row["case_id"] for row in evaluation],
        "final_holdout_case_ids": [],
    }
    _write(existing, package)
    _write(STUDY / "calibration_results.json", results)
    _write(
        STUDY / "calibration_summary.json",
        {
            "complete": True,
            "selected_key": selected["key"],
            "overall_metrics": selected["overall"],
            "regime_metrics": selected["by_regime"],
            "coverage_at_precision80": selected["frontier"]["selected"]["coverage"],
            "selected_threshold": selected["frontier"]["selected"]["threshold"],
            "overall_ece_target_met": selected["overall"]["ece"] < 0.05,
            "critical_regime_ece_target_met": all(
                value["ece"] < 0.10 for value in selected["by_regime"].values()
            ),
            "final_holdouts_used": False,
        },
    )
    print(existing)
    return existing


if __name__ == "__main__":
    run()
