#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
MODEL_STUDY = ROOT / "artifacts/studies/v4_model_selection"
VALIDATION = ROOT / "artifacts/studies/v4_precision_validation"
CONFIG = ROOT / "configs/v4/model_selection.yaml"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run() -> Path:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    model_path = ROOT / config["freeze_model"]
    threshold_path = ROOT / config["freeze_thresholds"]
    manifest_path = ROOT / config["freeze_manifest"]
    if model_path.is_file() or threshold_path.is_file():
        if not model_path.is_file() or not threshold_path.is_file() or not manifest_path.is_file():
            raise RuntimeError("partial v4 freeze state is invalid")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _sha(model_path) != manifest["frozen_model_hash"]:
            raise RuntimeError("frozen v4 model checksum changed")
        if _sha(threshold_path) != manifest["frozen_threshold_hash"]:
            raise RuntimeError("frozen v4 threshold checksum changed")
        print(threshold_path)
        return threshold_path
    if (ROOT / "artifacts/studies/v4_frozen_holdout/summary.json").is_file():
        raise RuntimeError("cannot freeze after v4 final holdout materialization")
    inputs = {
        "probability_package": VALIDATION / "probability_package.json",
        "benefit_package": VALIDATION / "benefit_package.json",
        "safety_package": VALIDATION / "safety_package.json",
        "threshold_selection": VALIDATION / "threshold_selection.json",
        "feature_schema": MODEL_STUDY / "feature_schema.json",
        "split_manifest": MODEL_STUDY / "split_manifest.json",
        "selected_base_model": MODEL_STUDY / "selected_base_model.json",
    }
    if any(not path.is_file() for path in inputs.values()):
        missing = [name for name, path in inputs.items() if not path.is_file()]
        raise RuntimeError(f"cannot freeze; missing development artifacts: {missing}")
    selection = json.loads(inputs["threshold_selection"].read_text(encoding="utf-8"))
    probability = json.loads(inputs["probability_package"].read_text(encoding="utf-8"))
    benefit = json.loads(inputs["benefit_package"].read_text(encoding="utf-8"))
    dataset = json.loads((MODEL_STUDY / "dataset_summary.json").read_text(encoding="utf-8"))
    model_payload = {
        "version": "concordia-v4-frozen-model-1",
        "frozen_before_final_holdout": True,
        "selected_policy": selection["selected_policy"],
        "deployment_allowed": selection["deployment_allowed"],
        "deployment_block_reason": selection["deployment_block_reason"],
        "selected_probability_model": json.loads(
            inputs["selected_base_model"].read_text(encoding="utf-8")
        )["name"],
        "calibration_method": probability["selected_calibration_method"],
        "benefit_mean_model": benefit["selected_mean_model"],
        "benefit_lower_model": benefit["selected_lower_model"],
        "artifact_paths": {
            name: str(path.relative_to(ROOT)) for name, path in inputs.items()
        },
        "artifact_hashes": {name: _sha(path) for name, path in inputs.items()},
        "feature_schema_hash": dataset["feature_schema_hash"],
        "data_split_hash": dataset["split_hash"],
        "final_holdout_case_ids": [],
        "rl_used": False,
    }
    threshold_payload = {
        "version": "concordia-v4-frozen-thresholds-1",
        "selected_policy": selection["selected_policy"],
        "deployment_allowed": selection["deployment_allowed"],
        "deployment_block_reason": selection["deployment_block_reason"],
        "final_operating_point": selection["selected_operating_point"],
        "policy_operating_points": selection["policy_operating_points"],
        "probability_threshold": selection["probability_threshold"],
        "benefit_threshold": selection["benefit_threshold"],
        "esiv_threshold": selection["esiv_threshold"],
        "safety_delta": selection["safety_delta"],
        "safety_failure_probability_threshold": selection[
            "safety_failure_probability_threshold"
        ],
        "minimum_relative_ttt_gain": 0.01,
        "regret_limit": 0.08,
        "precision_target": 0.80,
        "coverage_minimum": 0.20,
        "coverage_stretch": 0.25,
        "coverage_guard": 0.15,
        "minimum_intervention_count": 30,
        "target_intervention_count": 50,
        "strong_precision_ci_lower": 0.60,
        "very_strong_precision_ci_lower": 0.70,
        "outcome_rules": {
            "S_plus": {"precision": 0.80, "coverage": 0.25},
            "S": {"precision": 0.80, "coverage": 0.20},
            "P": {"precision": 0.70},
            "F": {"precision_below": 0.70},
        },
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    threshold_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(yaml.safe_dump(model_payload, sort_keys=False), encoding="utf-8")
    threshold_path.write_text(
        yaml.safe_dump(threshold_payload, sort_keys=False), encoding="utf-8"
    )
    manifest = {
        "complete": True,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "frozen_model_hash": _sha(model_path),
        "frozen_threshold_hash": _sha(threshold_path),
        "model_artifact_hashes": model_payload["artifact_hashes"],
        "feature_schema_hash": dataset["feature_schema_hash"],
        "data_split_hash": dataset["split_hash"],
        "final_holdout_started": False,
        "immutable": True,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(threshold_path)
    return threshold_path


if __name__ == "__main__":
    run()
