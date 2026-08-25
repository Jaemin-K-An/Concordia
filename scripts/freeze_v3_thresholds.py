#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "artifacts" / "studies" / "v3_feasibility_prediction"
CONFIG = ROOT / "configs" / "v3" / "model_selection.yaml"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run() -> Path:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    freeze_path = ROOT / config["freeze_file"]
    manifest_path = ROOT / config["freeze_manifest"]
    if freeze_path.is_file():
        if not manifest_path.is_file():
            raise RuntimeError("frozen thresholds exist without a freeze manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _sha(freeze_path) != manifest["threshold_config_hash"]:
            raise RuntimeError("frozen threshold checksum changed")
        print(freeze_path)
        return freeze_path
    if (ROOT / "artifacts/studies/v3_selective_holdout/summary.json").exists():
        raise RuntimeError("cannot freeze thresholds after holdout materialization")
    selected_path = STUDY / "selected_model.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    prereg = yaml.safe_load((ROOT / "configs/v3/preregistration.yaml").read_text())
    dataset = json.loads((STUDY / "raw_metrics.json").read_text(encoding="utf-8"))
    validation = [row for row in dataset if row["source_split"] == "validation"]
    overlap_values = np.asarray([row["features"]["route_overlap"] for row in validation])
    preference_values = np.asarray(
        [row["features"]["preference_variance"] for row in validation]
    )
    payload = {
        "version": "concordia-v3-frozen-1",
        "frozen_before_holdout": True,
        "p_win_threshold": float(
            selected["threshold_selection"]["selected"]["threshold"]
        ),
        "maximum_uncertainty": float(selected["maximum_uncertainty"]),
        "aps_threshold": float(
            selected["aps_threshold_selection"]["selected"]["threshold"]
        ),
        "maximum_route_overlap": float(np.median(overlap_values)),
        "minimum_preference_variance": float(np.median(preference_values)),
        "minimum_relative_ttt_gain": float(
            prereg["primary"]["minimum_relative_ttt_gain"]
        ),
        "safety_delta": float(prereg["success_definition"]["safety_delta"]),
        "minimum_acceptance_probability": float(
            selected["minimum_acceptance_probability"]
        ),
        "maximum_tail_loss": float(selected["maximum_tail_loss"]),
        "benefit_lcb_z": float(selected["benefit_lcb_z"]),
        "engineering_precision_target": float(
            prereg["primary"]["intervention_precision_target"]
        ),
        "coverage_target": float(prereg["primary"]["coverage_target"]),
        "scientific_precision_lower_ci_target": float(
            prereg["primary"]["scientific_precision_lower_ci_target"]
        ),
        "selected_model_name": selected["selected_model_name"],
        "selected_model_hash": _sha(selected_path),
        "feature_schema_hash": selected["feature_schema_hash"],
        "data_split_hash": selected["split_hash"],
    }
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "complete": True,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "threshold_config": str(freeze_path.relative_to(ROOT)),
        "threshold_config_hash": _sha(freeze_path),
        "selected_model_hash": _sha(selected_path),
        "feature_schema_hash": selected["feature_schema_hash"],
        "data_split_hash": selected["split_hash"],
        "holdout_started": False,
        "immutable": True,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(freeze_path)
    return freeze_path


if __name__ == "__main__":
    run()
