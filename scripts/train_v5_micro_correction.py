#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from concordia.feasibility import (
    MICRO_ADDITIONAL_FEATURES,
    MicroscopicCorrectionModel,
    ProbabilityCalibrator,
    V5_FEATURE_SCHEMA,
    unified_calibration_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "artifacts/studies/v5_micro_calibration"
MICRO_FEATURE_SCHEMA = V5_FEATURE_SCHEMA + MICRO_ADDITIONAL_FEATURES


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _matrix(rows: list[dict]) -> np.ndarray:
    return np.asarray(
        [[row["micro_features"][name] for name in MICRO_FEATURE_SCHEMA] for row in rows],
        dtype=float,
    )


def run() -> Path:
    existing = STUDY / "micro_correction_package.json"
    if (ROOT / "configs/v5/frozen_micro_safety.yaml").is_file():
        if not existing.is_file():
            raise RuntimeError("v5 is frozen but micro correction package is missing")
        print(existing)
        return existing
    rows = json.loads((STUDY / "raw_metrics.json").read_text())
    development = [row for row in rows if row["development_role"] == "micro_development"]
    calibration = [row for row in rows if row["development_role"] == "micro_calibration"]
    validation = [row for row in rows if row["development_role"] == "micro_validation"]
    model = MicroscopicCorrectionModel.fit(
        _matrix(development),
        [row["analytical_benefit"] for row in development],
        [row["microscopic_benefit"] for row in development],
        [row["microscopic_success"] for row in development],
        MICRO_FEATURE_SCHEMA,
    )
    _corrected_cal, raw_cal = model.predict(
        _matrix(calibration), [row["analytical_benefit"] for row in calibration]
    )
    y_cal = np.asarray([row["microscopic_success"] for row in calibration], dtype=int)
    methods = {}
    for method in ("platt", "isotonic", "beta"):
        calibrator = ProbabilityCalibrator(method).fit(raw_cal, y_cal)
        _corrected_val, raw_val = model.predict(
            _matrix(validation), [row["analytical_benefit"] for row in validation]
        )
        probability = calibrator.predict(raw_val)
        methods[method] = {
            "metrics": unified_calibration_metrics(
                [row["microscopic_success"] for row in validation], probability
            ),
            "calibrator": calibrator.to_dict(),
        }
    selected = min(
        methods,
        key=lambda method: (
            methods[method]["metrics"]["brier_score"],
            methods[method]["metrics"]["ece"],
        ),
    )
    corrected, raw = model.predict(
        _matrix(validation), [row["analytical_benefit"] for row in validation]
    )
    probability = ProbabilityCalibrator.from_dict(
        methods[selected]["calibrator"]
    ).predict(raw)
    realized = np.asarray([row["microscopic_benefit"] for row in validation])
    package = {
        "complete": True,
        "model": model.to_dict(),
        "calibrator": methods[selected]["calibrator"],
        "selected_calibration": selected,
        "feature_schema": list(MICRO_FEATURE_SCHEMA),
        "training_case_ids": [row["case_id"] for row in development],
        "calibration_case_ids": [row["case_id"] for row in calibration],
        "validation_case_ids": [row["case_id"] for row in validation],
        "final_holdout_case_ids": [],
    }
    _write(existing, package)
    _write(
        STUDY / "micro_correction_summary.json",
        {
            "complete": True,
            "selected_calibration": selected,
            "calibration_comparison": {
                method: values["metrics"] for method, values in methods.items()
            },
            "validation_correction_bias": float(np.mean(realized - corrected)),
            "validation_correction_mae": float(np.mean(np.abs(realized - corrected))),
            "validation_success_metrics": unified_calibration_metrics(
                [row["microscopic_success"] for row in validation], probability
            ),
            "final_holdout_used": False,
        },
    )
    print(existing)
    return existing


if __name__ == "__main__":
    run()
