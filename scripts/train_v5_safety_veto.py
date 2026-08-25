#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from concordia.feasibility import MICRO_ADDITIONAL_FEATURES, V5_FEATURE_SCHEMA
from concordia.safety import MicroscopicSafetyVeto


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/v5/microscopic.yaml"
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


def _evaluate(labels: np.ndarray, upper: np.ndarray, threshold: float) -> dict:
    safe = upper < threshold
    false_safe = int(np.sum(safe & (labels == 1)))
    return {
        "threshold": threshold,
        "predicted_safe_count": int(safe.sum()),
        "safe_coverage": float(safe.mean()),
        "false_safe_count": false_safe,
        "false_safe_rate": false_safe / max(1, len(labels)),
    }


def run() -> Path:
    existing = STUDY / "micro_safety_package.json"
    if (ROOT / "configs/v5/frozen_micro_safety.yaml").is_file():
        if not existing.is_file():
            raise RuntimeError("v5 is frozen but micro safety package is missing")
        print(existing)
        return existing
    config = yaml.safe_load(CONFIG.read_text())
    rows = json.loads((STUDY / "raw_metrics.json").read_text())
    development = [row for row in rows if row["development_role"] == "micro_development"]
    calibration = [row for row in rows if row["development_role"] == "micro_calibration"]
    validation = [row for row in rows if row["development_role"] == "micro_validation"]
    veto = MicroscopicSafetyVeto.fit(
        _matrix(development),
        [row["microscopic_safety_violation"] for row in development],
        MICRO_FEATURE_SCHEMA,
        seed=20260829,
    )
    _mean, upper = veto.predict(_matrix(calibration))
    labels = np.asarray(
        [row["microscopic_safety_violation"] for row in calibration], dtype=int
    )
    candidates = [
        _evaluate(labels, upper, float(threshold))
        for threshold in config["micro_safety_probability_thresholds"]
    ]
    feasible = [row for row in candidates if row["false_safe_rate"] <= 0.05]
    selected = max(
        feasible or candidates,
        key=lambda row: (
            row["false_safe_rate"] <= 0.05,
            row["safe_coverage"],
            -row["false_safe_rate"],
        ),
    )
    veto.probability_threshold = float(selected["threshold"])
    _mean_validation, upper_validation = veto.predict(_matrix(validation))
    validation_metrics = _evaluate(
        np.asarray(
            [row["microscopic_safety_violation"] for row in validation], dtype=int
        ),
        upper_validation,
        veto.probability_threshold,
    )
    package = {
        "complete": True,
        "veto": veto.to_dict(),
        "feature_schema": list(MICRO_FEATURE_SCHEMA),
        "threshold_selection": candidates,
        "selected_calibration_operating_point": selected,
        "validation_metrics": validation_metrics,
        "training_case_ids": [row["case_id"] for row in development],
        "calibration_case_ids": [row["case_id"] for row in calibration],
        "validation_case_ids": [row["case_id"] for row in validation],
        "final_holdout_case_ids": [],
    }
    _write(existing, package)
    _write(
        STUDY / "micro_safety_summary.json",
        {
            "complete": True,
            "selected_threshold": veto.probability_threshold,
            "calibration_metrics": selected,
            "validation_metrics": validation_metrics,
            "false_safe_target_met": validation_metrics["false_safe_rate"] <= 0.05,
            "final_holdout_used": False,
        },
    )
    print(existing)
    return existing


if __name__ == "__main__":
    run()
