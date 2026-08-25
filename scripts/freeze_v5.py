#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml

from v5_frozen import ROOT, sha256, write_json


FROZEN_CONFIGS = {
    "configs/v5/frozen_regimes.yaml": "regimes",
    "configs/v5/frozen_model.yaml": "model",
    "configs/v5/frozen_thresholds.yaml": "thresholds",
    "configs/v5/frozen_shift_detector.yaml": "shift_detector",
    "configs/v5/frozen_micro_safety.yaml": "micro_safety",
}

ARTIFACT_PATHS = [
    "artifacts/studies/v5_model_selection/feature_schema.json",
    "artifacts/studies/v5_model_selection/split_manifest.json",
    "artifacts/studies/v5_shift_detection/regime_definition.json",
    "artifacts/studies/v5_shift_detection/shift_detector.json",
    "artifacts/studies/v5_shift_detection/calibrated_model.json",
    "artifacts/studies/v5_micro_calibration/micro_correction_package.json",
    "artifacts/studies/v5_micro_calibration/micro_safety_package.json",
    "artifacts/studies/v5_policy_validation/threshold_package.json",
    "artifacts/studies/v5_policy_validation/validation_summary.json",
]

SCRIPT_PATHS = [
    "scripts/build_v5_dataset.py",
    "scripts/build_v5_micro_dataset.py",
    "scripts/run_microscopic_study.py",
    "scripts/run_real_topology_study.py",
    "scripts/run_v3_real_topology.py",
    "scripts/v5_frozen.py",
    "scripts/run_v5_holdout.py",
    "scripts/run_v5_microscopic.py",
    "scripts/run_v5_real_topology.py",
    "scripts/run_v5_stress.py",
]


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_yaml(path: Path, value) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=True), encoding="utf-8")


def run() -> Path:
    manifest_path = ROOT / "artifacts/v5/freeze_manifest.json"
    if manifest_path.is_file():
        print(manifest_path)
        return manifest_path
    if _git("status", "--porcelain"):
        raise RuntimeError("v5 freeze requires a clean committed source tree")
    forbidden = [
        ROOT / "artifacts/studies/v5_frozen_holdout",
        ROOT / "artifacts/studies/v5_microscopic_holdout",
        ROOT / "artifacts/studies/v5_stress_holdout",
        ROOT / "artifacts/studies/v5_real_topology",
    ]
    if any(path.exists() for path in forbidden):
        raise RuntimeError("a v5 final holdout was materialized before freeze")
    artifact_hashes = {
        relative: sha256(ROOT / relative) for relative in ARTIFACT_PATHS
    }
    threshold_package = json.loads(
        (ROOT / "artifacts/studies/v5_policy_validation/threshold_package.json").read_text()
    )
    regime_package = json.loads(
        (ROOT / "artifacts/studies/v5_shift_detection/regime_definition.json").read_text()
    )
    shift_package = json.loads(
        (ROOT / "artifacts/studies/v5_shift_detection/shift_detector.json").read_text()
    )
    analytical = json.loads(
        (ROOT / "artifacts/studies/v5_shift_detection/calibrated_model.json").read_text()
    )
    micro_safety = json.loads(
        (ROOT / "artifacts/studies/v5_micro_calibration/micro_safety_package.json").read_text()
    )
    common = {
        "version": "concordia-v5-frozen-1",
        "source_commit": _git("rev-parse", "HEAD"),
        "final_holdouts_used": False,
    }
    payloads = {
        "regimes": {
            **common,
            "definition": regime_package["definition"],
            "artifact_path": "artifacts/studies/v5_shift_detection/regime_definition.json",
            "artifact_hash": artifact_hashes[
                "artifacts/studies/v5_shift_detection/regime_definition.json"
            ],
        },
        "model": {
            **common,
            "selected_key": analytical["selected_key"],
            "selected_model": analytical["selected_model"],
            "selected_calibration": analytical["selected_calibration"],
            "artifact_path": "artifacts/studies/v5_shift_detection/calibrated_model.json",
            "artifact_hash": artifact_hashes[
                "artifacts/studies/v5_shift_detection/calibrated_model.json"
            ],
            "feature_schema_path": "artifacts/studies/v5_model_selection/feature_schema.json",
        },
        "thresholds": {**common, "thresholds": threshold_package},
        "shift_detector": {
            **common,
            "detector": shift_package["detector"],
            "feature_names": shift_package["feature_names"],
            "artifact_path": "artifacts/studies/v5_shift_detection/shift_detector.json",
            "artifact_hash": artifact_hashes[
                "artifacts/studies/v5_shift_detection/shift_detector.json"
            ],
        },
        "micro_safety": {
            **common,
            "probability_threshold": micro_safety["veto"]["probability_threshold"],
            "micro_success_threshold": threshold_package["micro_success_threshold"],
            "correction_artifact_path": "artifacts/studies/v5_micro_calibration/micro_correction_package.json",
            "safety_artifact_path": "artifacts/studies/v5_micro_calibration/micro_safety_package.json",
            "correction_artifact_hash": artifact_hashes[
                "artifacts/studies/v5_micro_calibration/micro_correction_package.json"
            ],
            "safety_artifact_hash": artifact_hashes[
                "artifacts/studies/v5_micro_calibration/micro_safety_package.json"
            ],
        },
    }
    for relative, key in FROZEN_CONFIGS.items():
        _write_yaml(ROOT / relative, payloads[key])
    source_paths = sorted(
        str(path.relative_to(ROOT))
        for path in (ROOT / "src/concordia").rglob("*.py")
    ) + SCRIPT_PATHS
    deployment_hashes = {
        relative: sha256(ROOT / relative) for relative in source_paths
    }
    frozen_hashes = {
        relative: sha256(ROOT / relative) for relative in FROZEN_CONFIGS
    }
    manifest = {
        "complete": True,
        "version": "concordia-v5-freeze-manifest-1",
        "source_commit": common["source_commit"],
        "source_clean_at_freeze": True,
        "frozen_config_hashes": frozen_hashes,
        "artifact_hashes": artifact_hashes,
        "deployment_code_hashes": deployment_hashes,
        "final_holdouts_materialized_at_freeze": False,
        "post_freeze_code_change_detection": True,
        "rl_used": False,
    }
    write_json(manifest_path, manifest)
    print(manifest_path)
    return manifest_path


if __name__ == "__main__":
    run()
