#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml

from concordia.v10.integrity import file_hashes
from concordia.v9.action_space import generate_action_library


ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "artifacts/studies/v10_racing_validation/validation_summary.json"
MANIFEST = ROOT / "artifacts/v10/freeze_manifest.json"
FINAL_SEEDS = ROOT / "artifacts/v10/final_seed_manifest.json"


def _write_yaml(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run() -> Path:
    if not VALIDATION.is_file():
        raise RuntimeError("v10 validation evidence is absent")
    if MANIFEST.exists() or FINAL_SEEDS.exists():
        raise RuntimeError("v10 freeze or final seed material already exists")
    validation = json.loads(VALIDATION.read_text())
    if not validation.get("freeze_authorized", False):
        raise RuntimeError("v10 validation did not authorize freeze")
    if validation.get("final_holdout_materialized", True):
        raise RuntimeError("v10 final holdout was materialized before freeze")
    racing_config_path = ROOT / validation["racing_config"]
    racing = yaml.safe_load(racing_config_path.read_text())
    action_space_path = ROOT / "configs/v10/frozen_action_space.yaml"
    stage1_path = ROOT / "configs/v10/frozen_stage1.yaml"
    stage2_path = ROOT / "configs/v10/frozen_stage2.yaml"
    stage3_path = ROOT / "configs/v10/frozen_stage3.yaml"
    selector_path = ROOT / "configs/v10/frozen_selector.yaml"
    thresholds_path = ROOT / "configs/v10/frozen_thresholds.yaml"
    _write_yaml(action_space_path, {
        "source": "concordia.v9.action_space.generate_action_library",
        "adaptive_action_count": 24,
        "null_action_always_available": True,
        "actions": [action.to_dict() for action in generate_action_library()],
    })
    _write_yaml(stage1_path, racing["stage_1"])
    _write_yaml(stage2_path, racing["stage_2"])
    _write_yaml(stage3_path, {
        **racing["stage_3"],
        "verification": racing["verification"],
    })
    _write_yaml(selector_path, racing["selector"])
    _write_yaml(thresholds_path, {
        "minimum_relative_benefit": 0.005,
        "maximum_risk_delta": 0.25,
        "maximum_regret": 0.08,
        "all_routes_legal_required": True,
        "validation_minimum_precision": 0.85,
        "final_minimum_precision": 0.80,
        "final_minimum_coverage": 0.10,
        "final_minimum_interventions": 40,
        "final_maximum_safety_violations": 0,
    })
    frozen_files = [
        action_space_path, stage1_path, stage2_path, stage3_path,
        selector_path, thresholds_path,
    ]
    implementation_files = [
        ROOT / "src/concordia/v9/action_space.py",
        ROOT / "src/concordia/v10/cache.py",
        ROOT / "src/concordia/v10/integrity.py",
        ROOT / "src/concordia/v10/racing.py",
        ROOT / "src/concordia/v10/seeds.py",
        ROOT / "src/concordia/v10/statistics.py",
        ROOT / "scripts/materialize_v10_final.py",
        ROOT / "scripts/v10_micro_sim.py",
        ROOT / "scripts/run_v10_racing_study.py",
        ROOT / "scripts/run_v10_final_holdout.py",
    ]
    missing = [str(path.relative_to(ROOT)) for path in implementation_files if not path.is_file()]
    if missing:
        raise RuntimeError(f"v10 freeze implementation is incomplete: {missing}")
    parent_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    _write_json(MANIFEST, {
        "study": "CONCORDIA_v10_multi_fidelity_action_racing",
        "created_from_validation_commit": parent_commit,
        "validation_summary": str(VALIDATION.relative_to(ROOT)),
        "validation_summary_sha256": file_hashes([VALIDATION], ROOT)[str(VALIDATION.relative_to(ROOT))],
        "selected_racing_config": str(racing_config_path.relative_to(ROOT)),
        "selected_racing_config_sha256": validation["racing_config_sha256"],
        "frozen_file_hashes": file_hashes(frozen_files, ROOT),
        "implementation_file_hashes": file_hashes(implementation_files, ROOT),
        "final_holdout_materialized": False,
        "rl_used": False,
    })
    print(MANIFEST)
    return MANIFEST


if __name__ == "__main__":
    run()
