#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml

from v7_frozen import FROZEN_CONFIGS, MANIFEST, ROOT, payload_hash, sha256, write_json


ARTIFACT_PATHS = (
    "artifacts/studies/v7_paired_dataset/raw_metrics.json",
    "artifacts/studies/v7_paired_dataset/dataset_summary.json",
    "artifacts/studies/v7_paired_dataset/feature_schema.json",
    "artifacts/studies/v7_paired_dataset/split_manifest.json",
    "artifacts/studies/v7_model_selection/candidate_comparison.json",
    "artifacts/studies/v7_model_selection/direct_quantile_models.json",
    "artifacts/studies/v7_model_selection/auxiliary_comparison.json",
    "artifacts/studies/v7_model_selection/effect_calibration.json",
    "artifacts/studies/v7_model_selection/cumulative_gain.json",
    "artifacts/studies/v7_model_selection/placebo.json",
    "artifacts/studies/v7_model_selection/ablation.json",
    "artifacts/studies/v7_model_selection/training_manifest.json",
    "artifacts/studies/v7_model_selection/scenario_family_holdouts.json",
    "artifacts/studies/v7_policy_validation/precision_coverage_frontier.json",
    "artifacts/studies/v7_policy_validation/policy_comparison.json",
    "artifacts/studies/v7_policy_validation/scenario_grouped_validation.json",
    "artifacts/studies/v7_policy_validation/selected_policy_package.json",
    "artifacts/studies/v7_policy_validation/validation_summary.json",
)

DEPLOYMENT_SCRIPTS = (
    "scripts/v7_micro_sim.py",
    "scripts/v7_frozen.py",
    "scripts/run_v7_analytical_holdout.py",
    "scripts/run_v7_microscopic_holdout.py",
    "scripts/run_v7_real_topology.py",
)


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _write_yaml(path: Path, value) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=True), encoding="utf-8")


def run() -> Path:
    if MANIFEST.is_file():
        from v7_frozen import verify_frozen

        verify_frozen()
        print(MANIFEST)
        return MANIFEST
    if _git("status", "--porcelain"):
        raise RuntimeError("v7 freeze requires a clean committed source and artifact tree")
    forbidden = (
        ROOT / "artifacts/studies/v7_frozen_analytical_holdout",
        ROOT / "artifacts/studies/v7_frozen_micro_holdout",
        ROOT / "artifacts/studies/v7_real_topology",
    )
    if any(path.exists() for path in forbidden):
        raise RuntimeError("a v7 final holdout was materialized before freeze")
    missing = [relative for relative in ARTIFACT_PATHS if not (ROOT / relative).is_file()]
    if missing:
        raise RuntimeError(f"v7 development artifact missing: {missing}")
    package = json.loads(
        (ROOT / "artifacts/studies/v7_policy_validation/selected_policy_package.json").read_text()
    )
    policy = package["policy"]
    common = {
        "version": "concordia-v7-frozen-1",
        "source_commit": _git("rev-parse", "HEAD"),
        "development_only_selection": True,
        "final_holdouts_used": False,
    }
    payloads = (
        {
            **common,
            "traffic_model": policy["traffic_model"],
            "traffic_bootstrap": policy["traffic_bootstrap"],
            "feature_schema": policy["feature_schema"],
        },
        {
            **common,
            "safety_model": policy["safety_model"],
            "safety_bootstrap": policy["safety_bootstrap"],
        },
        {
            **common,
            "regret_model": policy["regret_model"],
            "regret_bootstrap": policy["regret_bootstrap"],
        },
        {
            **common,
            "conformal": policy["conformal"],
            "bootstrap_interval_quantiles": [0.10, 0.90],
        },
        {
            **common,
            "interval_method": policy["interval_method"],
            "traffic_lcb_threshold": policy["traffic_lcb_threshold"],
            "safety_ucb_threshold": policy["safety_ucb_threshold"],
            "regret_ucb_threshold": policy["regret_ucb_threshold"],
            "safe_abstention": bool(package["safe_abstention"]),
            "selection": package["selection"],
        },
    )
    for relative, payload in zip(FROZEN_CONFIGS, payloads):
        _write_yaml(ROOT / relative, payload)
    source_paths = sorted(
        str(path.relative_to(ROOT))
        for path in (ROOT / "src/concordia/uplift_v7").glob("*.py")
    ) + list(DEPLOYMENT_SCRIPTS)
    missing_source = [relative for relative in source_paths if not (ROOT / relative).is_file()]
    if missing_source:
        raise RuntimeError(f"v7 deployment source missing: {missing_source}")
    v6_manifest = ROOT / "artifacts/v6/freeze_manifest.json"
    manifest = {
        "complete": True,
        "version": "concordia-v7-freeze-manifest-1",
        "source_commit": common["source_commit"],
        "source_clean_at_freeze": True,
        "frozen_config_hashes": {
            relative: sha256(ROOT / relative) for relative in FROZEN_CONFIGS
        },
        "artifact_hashes": {
            relative: sha256(ROOT / relative) for relative in ARTIFACT_PATHS
        },
        "deployment_code_hashes": {
            relative: sha256(ROOT / relative) for relative in source_paths
        },
        "upstream_v6_manifest_hash": sha256(v6_manifest),
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
