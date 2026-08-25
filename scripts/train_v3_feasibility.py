#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from concordia.feasibility import FEATURE_SCHEMA, FeasibilityModel, build_candidate_models


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "artifacts" / "studies" / "v3_feasibility_prediction"


def _write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ridge(matrix: np.ndarray, target: np.ndarray, penalty: float = 1e-3) -> dict:
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-10] = 1.0
    normalized = (matrix - mean) / scale
    design = np.column_stack([np.ones(len(normalized)), normalized])
    regularizer = np.eye(design.shape[1]) * penalty
    regularizer[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + regularizer, design.T @ target)
    prediction = design @ coefficients
    return {
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "intercept": float(coefficients[0]),
        "coefficients": coefficients[1:].tolist(),
        "residual_standard_deviation": float(np.std(target - prediction, ddof=1)),
        "training_r_squared": float(
            1.0 - np.sum((target - prediction) ** 2) / max(np.sum((target - target.mean()) ** 2), 1e-12)
        ),
    }


def run() -> Path:
    config = yaml.safe_load((ROOT / "configs/v3/model_selection.yaml").read_text())
    split = yaml.safe_load((ROOT / "configs/v3/splits.yaml").read_text())
    rows = json.loads((STUDY / "raw_metrics.json").read_text(encoding="utf-8"))
    train = [row for row in rows if row["source_split"] == "training"]
    if any(row["seed"] in split["holdout"]["seeds"] for row in train):
        raise RuntimeError("holdout seed entered model fitting")
    matrix = np.asarray(
        [[row["features"][name] for name in FEATURE_SCHEMA] for row in train], dtype=float
    )
    labels = np.asarray([int(row["label"] == "WIN") for row in train], dtype=int)
    candidates = build_candidate_models(FEATURE_SCHEMA, int(config["bootstrap_seed"]))
    fitted = [model.fit(matrix, labels) for model in candidates]
    multiclass = {}
    for label_name in ("WIN", "TRADEOFF", "INFEASIBLE"):
        binary = np.asarray([int(row["label"] == label_name) for row in train], dtype=int)
        if len(np.unique(binary)) == 2:
            multiclass[label_name] = FeasibilityModel(
                f"secondary_{label_name}",
                "logistic",
                FEATURE_SCHEMA,
                regularization=0.03,
                seed=int(config["bootstrap_seed"]) + len(multiclass) + 100,
            ).fit(matrix, binary).to_dict()
    benefit_target = np.asarray(
        [row["adaptive_counterfactual"]["relative_ttt_gain"] for row in train],
        dtype=float,
    )
    safety_target = np.asarray(
        [row["adaptive_counterfactual"]["safety_difference"] for row in train],
        dtype=float,
    )
    payload = {
        "complete": True,
        "feature_names": list(FEATURE_SCHEMA),
        "training_case_ids": [row["case_id"] for row in train],
        "training_seeds": sorted({row["seed"] for row in train}),
        "training_scenarios": sorted({row["scenario"] for row in train}),
        "candidate_models": [model.to_dict() for model in fitted],
        "secondary_multiclass_models": multiclass,
        "benefit_regression": _ridge(matrix, benefit_target),
        "safety_regression": _ridge(matrix, safety_target),
        "split_hash": json.loads((STUDY / "dataset_summary.json").read_text())["split_hash"],
        "feature_schema_hash": _sha(STUDY / "feature_schema.json"),
        "holdout_case_ids": [],
        "holdout_used": False,
    }
    output = STUDY / "trained_candidates.json"
    _write(output, payload)
    _write(
        STUDY / "training_summary.json",
        {
            "complete": True,
            "model_count": len(fitted),
            "training_count": len(train),
            "win_count": int(labels.sum()),
            "nonwin_count": int(len(labels) - labels.sum()),
            "benefit_regression_r_squared": payload["benefit_regression"]["training_r_squared"],
            "safety_regression_r_squared": payload["safety_regression"]["training_r_squared"],
            "model_package_hash": _sha(output),
        },
    )
    print(output)
    return output


if __name__ == "__main__":
    run()
