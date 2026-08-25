#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from concordia.feasibility import (
    SafetyPredictionModel,
    V4_FEATURE_SCHEMA,
    false_safe_rate,
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


def run() -> Path:
    existing = STUDY / "safety_package.json"
    if (ROOT / "configs/v4/frozen_model.yaml").is_file():
        if not existing.is_file():
            raise RuntimeError("v4 is frozen but its safety package is missing")
        print(existing)
        return existing
    prereg = yaml.safe_load(
        (ROOT / "configs/v4/preregistration.yaml").read_text(encoding="utf-8")
    )
    config = yaml.safe_load(
        (ROOT / "configs/v4/model_selection.yaml").read_text(encoding="utf-8")
    )
    delta = float(prereg["success_definition"]["safety_delta"])
    rows = json.loads((MODEL_STUDY / "raw_metrics.json").read_text(encoding="utf-8"))
    fit = [
        row
        for row in rows
        if row["development_role"] in {"training", "calibration_fit"}
    ]
    evaluation = [
        row for row in rows if row["development_role"] == "calibration_evaluation"
    ]
    initial = SafetyPredictionModel.fit(
        _matrix(fit),
        np.asarray([row["targets"]["safety_difference"] for row in fit]),
        V4_FEATURE_SCHEMA,
        delta,
        int(config["bootstrap_seed"]) + 500,
    )
    predicted_upper, predicted_probability = initial.predict(_matrix(evaluation))
    actual = np.asarray([row["targets"]["safety_difference"] for row in evaluation])
    adjustment = max(0.0, float(np.max(actual - predicted_upper))) if len(actual) else 0.0
    adjusted_upper = predicted_upper + adjustment
    evaluation_metrics = false_safe_rate(actual, adjusted_upper, delta)
    final_rows = [row for row in rows if row["development_role"] != "validation"]
    final = SafetyPredictionModel.fit(
        _matrix(final_rows),
        np.asarray([row["targets"]["safety_difference"] for row in final_rows]),
        V4_FEATURE_SCHEMA,
        delta,
        int(config["bootstrap_seed"]) + 700,
    )
    package = {
        "complete": True,
        "model": final.to_dict(),
        "upper_adjustment": adjustment,
        "probability_upper_buffer": float(config["safety_probability_upper_buffer"]),
        "safety_failure_probability_threshold": float(
            config["safety_failure_probability_threshold"]
        ),
        "evaluation_metrics": evaluation_metrics,
        "evaluation_probabilities": predicted_probability.tolist(),
        "training_case_ids": [row["case_id"] for row in final_rows],
        "validation_case_ids": [],
        "final_holdout_case_ids": [],
    }
    output = STUDY / "safety_package.json"
    _write(output, package)
    _write(
        STUDY / "safety_summary.json",
        {
            "complete": True,
            "calibration_false_safe_metrics": evaluation_metrics,
            "upper_adjustment": adjustment,
            "final_holdout_used": False,
        },
    )
    print(output)
    return output


if __name__ == "__main__":
    run()
