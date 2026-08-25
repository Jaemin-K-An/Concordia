#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from concordia.feasibility import RobustShiftDetector, V5_FEATURE_SCHEMA


ROOT = Path(__file__).resolve().parents[1]
MODEL_STUDY = ROOT / "artifacts/studies/v5_model_selection"
STUDY = ROOT / "artifacts/studies/v5_shift_detection"
SHIFT_FEATURES = V5_FEATURE_SCHEMA[:-1]


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _matrix(rows: list[dict]) -> np.ndarray:
    return np.asarray(
        [[row["features"][name] for name in SHIFT_FEATURES] for row in rows], dtype=float
    )


def run() -> Path:
    existing = STUDY / "shift_detector.json"
    if (ROOT / "configs/v5/frozen_shift_detector.yaml").is_file():
        if not existing.is_file():
            raise RuntimeError("v5 shift detector is frozen but package is missing")
        print(existing)
        return existing
    config = yaml.safe_load((ROOT / "configs/v5/model.yaml").read_text())
    rows = json.loads((MODEL_STUDY / "regime_rows.json").read_text())
    training = [row for row in rows if row["development_role"] == "training"]
    detector = RobustShiftDetector.fit(
        _matrix(training),
        mild_quantile=float(config["shift_mild_quantile"]),
        strong_quantile=float(config["shift_strong_quantile"]),
    )
    scores = detector.score(_matrix(rows))
    classes = detector.classify(_matrix(rows))
    enriched = []
    by_role = defaultdict(list)
    for row, score, shift_class in zip(rows, scores, classes):
        value = dict(row)
        value["features"] = dict(row["features"])
        value["features"]["dss_penetration_interaction"] = float(score) * float(
            value["features"]["navigation_penetration"]
        )
        value["domain_shift_score"] = float(score)
        value["shift_class"] = shift_class
        enriched.append(value)
        by_role[value["development_role"]].append(float(score))
    package = {
        "complete": True,
        "detector": detector.to_dict(),
        "feature_names": list(SHIFT_FEATURES),
        "fit_roles": ["training"],
        "score_by_role": {
            role: {
                "mean": float(np.mean(values)),
                "p95": float(np.percentile(values, 95)),
            }
            for role, values in by_role.items()
        },
        "strong_shift_action": config["strong_shift_action"],
        "final_holdouts_used": False,
    }
    _write(existing, package)
    _write(MODEL_STUDY / "enriched_rows.json", enriched)
    print(existing)
    return existing


if __name__ == "__main__":
    run()
