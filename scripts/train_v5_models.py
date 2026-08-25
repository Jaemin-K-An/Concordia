#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from concordia.feasibility import BenefitModel, V5SuccessModel, V5_FEATURE_SCHEMA


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "artifacts/studies/v5_model_selection"


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _matrix(rows: list[dict]) -> np.ndarray:
    return np.asarray(
        [[row["features"][name] for name in V5_FEATURE_SCHEMA] for row in rows],
        dtype=float,
    )


def run() -> Path:
    existing = STUDY / "trained_candidates.json"
    if (ROOT / "configs/v5/frozen_model.yaml").is_file():
        if not existing.is_file():
            raise RuntimeError("v5 is frozen but trained candidates are missing")
        print(existing)
        return existing
    config = yaml.safe_load((ROOT / "configs/v5/model.yaml").read_text())
    rows = json.loads((STUDY / "regime_rows.json").read_text())
    training = [row for row in rows if row["development_role"] == "training"]
    matrix = _matrix(training)
    labels = np.asarray([row["targets"]["success"] for row in training], dtype=int)
    regimes = [row["regime"] for row in training]
    candidates = [
        V5SuccessModel.fit(
            name,
            matrix,
            labels,
            regimes,
            V5_FEATURE_SCHEMA,
            pooling_strength=float(config["hierarchical_pooling_strength"]),
        )
        for name in config["candidate_models"]
    ]
    benefit = BenefitModel(
        "v5_benefit_gradient_boosting", "boosting", iterations=100, learning_rate=0.06
    ).fit(
        matrix,
        np.asarray([row["targets"]["relative_ttt_gain"] for row in training]),
    )
    package = {
        "complete": True,
        "candidate_models": [model.to_dict() for model in candidates],
        "benefit_model": benefit.to_dict(),
        "feature_schema": list(V5_FEATURE_SCHEMA),
        "training_case_ids": [row["case_id"] for row in training],
        "final_holdout_case_ids": [],
    }
    _write(existing, package)
    _write(
        STUDY / "training_summary.json",
        {
            "complete": True,
            "training_count": len(training),
            "candidate_names": [model.name for model in candidates],
            "success_prevalence": float(labels.mean()),
            "final_holdouts_used": False,
        },
    )
    print(existing)
    return existing


if __name__ == "__main__":
    run()
