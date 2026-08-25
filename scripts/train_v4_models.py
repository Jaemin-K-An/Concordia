#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from concordia.feasibility import V4_FEATURE_SCHEMA, build_v4_candidate_models


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "artifacts/studies/v4_model_selection"


def _write(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run() -> Path:
    config = yaml.safe_load(
        (ROOT / "configs/v4/model_selection.yaml").read_text(encoding="utf-8")
    )
    rows = json.loads((STUDY / "raw_metrics.json").read_text(encoding="utf-8"))
    train = [row for row in rows if row["development_role"] == "training"]
    matrix = np.asarray(
        [[row["features"][name] for name in V4_FEATURE_SCHEMA] for row in train],
        dtype=float,
    )
    labels = np.asarray([row["targets"]["success"] for row in train], dtype=int)
    models = build_v4_candidate_models(V4_FEATURE_SCHEMA, int(config["bootstrap_seed"]))
    fitted = [model.fit(matrix, labels) for model in models]
    payload = {
        "complete": True,
        "feature_names": list(V4_FEATURE_SCHEMA),
        "training_case_ids": [row["case_id"] for row in train],
        "training_seeds": sorted({row["seed"] for row in train}),
        "training_count": len(train),
        "win_count": int(labels.sum()),
        "nonwin_count": int(len(labels) - labels.sum()),
        "candidate_models": [model.to_dict() for model in fitted],
        "final_holdout_case_ids": [],
        "final_holdout_used": False,
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
            "final_holdout_used": False,
        },
    )
    print(output)
    return output


if __name__ == "__main__":
    run()
