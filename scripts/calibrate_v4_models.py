#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/concordia-matplotlib-v4")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/concordia-cache-v4")

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from concordia.feasibility import (
    ConformalRiskController,
    ProbabilityCalibrator,
    V4BootstrapEnsemble,
    V4_FEATURE_SCHEMA,
    V4ProbabilityModel,
    calibration_error,
)


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "artifacts/studies/v4_precision_validation"
MODEL_STUDY = ROOT / "artifacts/studies/v4_model_selection"


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


def run() -> Path:
    existing = STUDY / "probability_package.json"
    if (ROOT / "configs/v4/frozen_model.yaml").is_file():
        if not existing.is_file():
            raise RuntimeError("v4 is frozen but its probability package is missing")
        print(existing)
        return existing
    config = yaml.safe_load(
        (ROOT / "configs/v4/model_selection.yaml").read_text(encoding="utf-8")
    )
    prereg = yaml.safe_load(
        (ROOT / "configs/v4/preregistration.yaml").read_text(encoding="utf-8")
    )
    rows = json.loads((MODEL_STUDY / "raw_metrics.json").read_text(encoding="utf-8"))
    train = [row for row in rows if row["development_role"] == "training"]
    calibration_fit = [
        row for row in rows if row["development_role"] == "calibration_fit"
    ]
    calibration_evaluation = [
        row for row in rows if row["development_role"] == "calibration_evaluation"
    ]
    prototype = V4ProbabilityModel.from_dict(
        json.loads((MODEL_STUDY / "selected_base_model.json").read_text(encoding="utf-8"))
    )
    train_x = _matrix(train)
    train_y = np.asarray([row["targets"]["success"] for row in train], dtype=int)
    groups = [f"{row['scenario']}-s{row['seed']}" for row in train]
    ensemble = V4BootstrapEnsemble.fit(
        prototype,
        train_x,
        train_y,
        groups,
        ensemble_size=int(config["bootstrap_ensemble_size"]),
        seed=int(config["bootstrap_seed"]),
    )
    fit_x = _matrix(calibration_fit)
    fit_y = np.asarray([row["targets"]["success"] for row in calibration_fit], dtype=int)
    evaluation_x = _matrix(calibration_evaluation)
    evaluation_y = np.asarray(
        [row["targets"]["success"] for row in calibration_evaluation], dtype=int
    )
    fit_raw, _fit_uncertainty, _fit_lower = ensemble.predict(fit_x)
    evaluation_raw, evaluation_uncertainty, evaluation_lower = ensemble.predict(evaluation_x)
    results = {}
    fitted = {}
    for method in config["calibration_methods"]:
        calibrator = ProbabilityCalibrator(str(method)).fit(fit_raw, fit_y)
        probabilities = calibrator.predict(evaluation_raw)
        results[method] = calibration_error(evaluation_y, probabilities)
        fitted[method] = calibrator
    ece_target = float(prereg["secondary"]["validation_ece_target"])
    selected_method = min(
        results,
        key=lambda method: (
            results[method]["ece"] > ece_target,
            results[method]["ece"],
            results[method]["brier_score"],
        ),
    )
    combined = calibration_fit + calibration_evaluation
    combined_x = _matrix(combined)
    combined_y = np.asarray([row["targets"]["success"] for row in combined], dtype=int)
    combined_raw, combined_uncertainty, combined_lower = ensemble.predict(combined_x)
    final_calibrator = ProbabilityCalibrator(selected_method).fit(combined_raw, combined_y)
    combined_calibrated = final_calibrator.predict(combined_raw)
    conformal = ConformalRiskController.fit(
        combined_calibrated,
        combined_y,
        target_error_rate=float(config["conformal_target_error"]),
        thresholds=config["probability_thresholds"],
    )
    package = {
        "complete": True,
        "ensemble": ensemble.to_dict(),
        "calibrator": final_calibrator.to_dict(),
        "selected_calibration_method": selected_method,
        "calibration_comparison": results,
        "calibration_fit_case_ids": [row["case_id"] for row in calibration_fit],
        "calibration_evaluation_case_ids": [
            row["case_id"] for row in calibration_evaluation
        ],
        "conformal_controller": conformal.to_dict(),
        "uncertainty_contract": {
            "lower_quantile": 0.10,
            "success_probability_lower_z": config["success_probability_lower_z"],
        },
        "final_holdout_case_ids": [],
        "final_holdout_used": False,
    }
    STUDY.mkdir(parents=True, exist_ok=True)
    output = STUDY / "probability_package.json"
    _write(output, package)
    _write(
        STUDY / "calibration_results.json",
        {
            "methods": results,
            "selected": selected_method,
            "evaluation_raw": evaluation_raw.tolist(),
            "evaluation_uncertainty": evaluation_uncertainty.tolist(),
            "evaluation_lower": evaluation_lower.tolist(),
            "evaluation_labels": evaluation_y.tolist(),
            "combined_uncertainty": combined_uncertainty.tolist(),
            "combined_lower": combined_lower.tolist(),
        },
    )
    figure_dir = STUDY / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(5.4, 5.0))
    for method, metrics in results.items():
        curve = metrics["curve"]
        axis.plot(
            [row["mean_probability"] for row in curve],
            [row["observed_rate"] for row in curve],
            marker="o",
            label=f"{method} (ECE {metrics['ece']:.3f})",
        )
    axis.plot([0, 1], [0, 1], linestyle="--", color="#888888")
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted P(success)", ylabel="Observed success")
    axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figure_dir / "calibration_comparison.png", dpi=180)
    plt.close(fig)
    _write(
        STUDY / "calibration_summary.json",
        {
            "complete": True,
            "selected_method": selected_method,
            "selected_ece": results[selected_method]["ece"],
            "selected_brier_score": results[selected_method]["brier_score"],
            "ece_target": ece_target,
            "ece_target_met": results[selected_method]["ece"] < ece_target,
            "final_holdout_used": False,
        },
    )
    print(output)
    return output


if __name__ == "__main__":
    run()
