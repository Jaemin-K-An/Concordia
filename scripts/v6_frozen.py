from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from concordia.micro_v6.policy import V6Policy


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifacts/v6/freeze_manifest.json"
FROZEN_CONFIGS = (
    "configs/v6/frozen_analytical_screener.yaml",
    "configs/v6/frozen_micro_model.yaml",
    "configs/v6/frozen_micro_calibration.yaml",
    "configs/v6/frozen_safety_model.yaml",
    "configs/v6/frozen_thresholds.yaml",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload_hash(value: dict) -> str:
    payload = {key: item for key, item in value.items() if key != "manifest_self_hash"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def verify_frozen() -> dict:
    if not MANIFEST.is_file():
        raise RuntimeError("v6 final evaluation is forbidden before freeze")
    manifest = json.loads(MANIFEST.read_text())
    if payload_hash(manifest) != manifest.get("manifest_self_hash"):
        raise RuntimeError("v6 freeze manifest changed")
    for section in ("frozen_config_hashes", "artifact_hashes", "deployment_code_hashes"):
        for relative, expected in manifest[section].items():
            path = ROOT / relative
            if not path.is_file() or sha256(path) != expected:
                raise RuntimeError(f"post-freeze v6 change detected: {relative}")
    return manifest


def load_policy() -> V6Policy:
    verify_frozen()
    analytical = yaml.safe_load((ROOT / FROZEN_CONFIGS[0]).read_text())
    model = yaml.safe_load((ROOT / FROZEN_CONFIGS[1]).read_text())
    calibration = yaml.safe_load((ROOT / FROZEN_CONFIGS[2]).read_text())
    safety = yaml.safe_load((ROOT / FROZEN_CONFIGS[3]).read_text())
    thresholds = yaml.safe_load((ROOT / FROZEN_CONFIGS[4]).read_text())
    return V6Policy.from_dict(
        {
            "composite": model["composite"],
            "benefit_model": model["benefit_model"],
            "composite_calibration": calibration["composite_calibration"],
            "benefit_calibration": calibration["benefit_calibration"],
            "safety_model": safety["safety_model"],
            "safety_calibration": safety["safety_calibration"],
            "architecture": thresholds["architecture"],
            "success_threshold": thresholds["success_threshold"],
            "safety_threshold": thresholds["safety_threshold"],
            "stage1_threshold": analytical["stage1_probability_threshold"],
            "conformal": thresholds["conformal"],
        }
    )

