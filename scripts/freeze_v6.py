#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml

from v6_frozen import FROZEN_CONFIGS, MANIFEST, ROOT, payload_hash, sha256, write_json


ARTIFACT_PATHS = (
    "artifacts/studies/v6_micro_dataset/raw_metrics.json",
    "artifacts/studies/v6_micro_dataset/dataset_summary.json",
    "artifacts/studies/v6_micro_dataset/feature_schema.json",
    "artifacts/studies/v6_micro_dataset/split_manifest.json",
    "artifacts/studies/v6_micro_model_selection/candidate_comparison.json",
    "artifacts/studies/v6_micro_model_selection/training_manifest.json",
    "artifacts/studies/v6_micro_model_selection/penetration_analysis.json",
    "artifacts/studies/v6_micro_calibration/calibration_comparison.json",
    "artifacts/studies/v6_policy_validation/precision_coverage_frontier.json",
    "artifacts/studies/v6_policy_validation/stage1_screening_frontier.json",
    "artifacts/studies/v6_policy_validation/ablation.json",
    "artifacts/studies/v6_policy_validation/selected_policy_package.json",
    "artifacts/studies/v6_policy_validation/threshold_package.json",
    "artifacts/studies/v6_policy_validation/validation_summary.json",
)

DEPLOYMENT_SCRIPTS = (
    "scripts/v6_micro_sim.py",
    "scripts/v6_frozen.py",
    "scripts/run_v6_analytical_holdout.py",
    "scripts/run_v6_microscopic_holdout.py",
    "scripts/run_v6_real_topology.py",
)


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _write_yaml(path: Path, value) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=True), encoding="utf-8")


def run() -> Path:
    if MANIFEST.is_file():
        from v6_frozen import verify_frozen

        verify_frozen()
        print(MANIFEST)
        return MANIFEST
    if _git("status", "--porcelain"):
        raise RuntimeError("v6 freeze requires a clean committed source and artifact tree")
    forbidden = (
        ROOT / "artifacts/studies/v6_frozen_analytical_holdout",
        ROOT / "artifacts/studies/v6_frozen_micro_holdout",
        ROOT / "artifacts/studies/v6_real_topology",
    )
    if any(path.exists() for path in forbidden):
        raise RuntimeError("a v6 final holdout was materialized before freeze")
    missing = [relative for relative in ARTIFACT_PATHS if not (ROOT / relative).is_file()]
    if missing:
        raise RuntimeError(f"v6 development artifact missing: {missing}")
    package = json.loads(
        (ROOT / "artifacts/studies/v6_policy_validation/selected_policy_package.json").read_text()
    )
    threshold_package = json.loads(
        (ROOT / "artifacts/studies/v6_policy_validation/threshold_package.json").read_text()
    )
    common = {
        "version": "concordia-v6-frozen-1",
        "source_commit": _git("rev-parse", "HEAD"),
        "final_holdouts_used": False,
    }
    v5_manifest = ROOT / "artifacts/v5/freeze_manifest.json"
    payloads = (
        {
            **common,
            "stage1_probability_threshold": package["stage1_threshold"],
            "upstream_analytical_model": "frozen_v5_model_used_as_recall_feature",
            "upstream_v5_manifest_hash": sha256(v5_manifest),
        },
        {
            **common,
            "feature_schema": package["feature_schema"],
            "composite": package["composite"],
            "benefit_model": package["benefit_model"],
        },
        {
            **common,
            "composite_calibration": package["composite_calibration"],
            "benefit_calibration": package["benefit_calibration"],
        },
        {
            **common,
            "safety_model": package["safety_model"],
            "safety_calibration": package["safety_calibration"],
        },
        {
            **common,
            "architecture": package["architecture"],
            "success_threshold": package["success_threshold"],
            "safety_threshold": package["safety_threshold"],
            "conformal": package["conformal"],
            "validation_selection": threshold_package,
        },
    )
    for relative, payload in zip(FROZEN_CONFIGS, payloads):
        _write_yaml(ROOT / relative, payload)
    source_paths = sorted(
        str(path.relative_to(ROOT))
        for path in (ROOT / "src/concordia/micro_v6").glob("*.py")
    ) + list(DEPLOYMENT_SCRIPTS)
    missing_source = [relative for relative in source_paths if not (ROOT / relative).is_file()]
    if missing_source:
        raise RuntimeError(f"v6 deployment source missing: {missing_source}")
    manifest = {
        "complete": True,
        "version": "concordia-v6-freeze-manifest-1",
        "source_commit": common["source_commit"],
        "source_clean_at_freeze": True,
        "frozen_config_hashes": {relative: sha256(ROOT / relative) for relative in FROZEN_CONFIGS},
        "artifact_hashes": {relative: sha256(ROOT / relative) for relative in ARTIFACT_PATHS},
        "deployment_code_hashes": {relative: sha256(ROOT / relative) for relative in source_paths},
        "final_holdouts_materialized_at_freeze": False,
        "post_freeze_code_change_detection": True,
        "rl_used": False,
    }
    manifest["manifest_self_hash"] = payload_hash(manifest)
    write_json(MANIFEST, manifest)
    print(MANIFEST)
    return MANIFEST


if __name__ == "__main__":
    run()
