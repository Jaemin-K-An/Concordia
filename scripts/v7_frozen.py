from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from concordia.uplift_v7.policy import UpliftPolicy


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifacts/v7/freeze_manifest.json"
FROZEN_CONFIGS = (
    "configs/v7/frozen_traffic_uplift.yaml",
    "configs/v7/frozen_safety_uplift.yaml",
    "configs/v7/frozen_regret_model.yaml",
    "configs/v7/frozen_intervals.yaml",
    "configs/v7/frozen_thresholds.yaml",
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
        raise RuntimeError("v7 final evaluation is forbidden before freeze")
    manifest = json.loads(MANIFEST.read_text())
    if payload_hash(manifest) != manifest.get("manifest_self_hash"):
        raise RuntimeError("v7 freeze manifest changed")
    for section in ("frozen_config_hashes", "artifact_hashes", "deployment_code_hashes"):
        for relative, expected in manifest[section].items():
            path = ROOT / relative
            if not path.is_file() or sha256(path) != expected:
                raise RuntimeError(f"post-freeze v7 change detected: {relative}")
    return manifest


def load_policy() -> UpliftPolicy:
    verify_frozen()
    traffic = yaml.safe_load((ROOT / FROZEN_CONFIGS[0]).read_text())
    safety = yaml.safe_load((ROOT / FROZEN_CONFIGS[1]).read_text())
    regret = yaml.safe_load((ROOT / FROZEN_CONFIGS[2]).read_text())
    intervals = yaml.safe_load((ROOT / FROZEN_CONFIGS[3]).read_text())
    thresholds = yaml.safe_load((ROOT / FROZEN_CONFIGS[4]).read_text())
    return UpliftPolicy.from_dict(
        {
            "traffic_model": traffic["traffic_model"],
            "traffic_bootstrap": traffic["traffic_bootstrap"],
            "conformal": intervals["conformal"],
            "safety_model": safety["safety_model"],
            "safety_bootstrap": safety["safety_bootstrap"],
            "regret_model": regret["regret_model"],
            "regret_bootstrap": regret["regret_bootstrap"],
            "interval_method": thresholds["interval_method"],
            "traffic_lcb_threshold": thresholds["traffic_lcb_threshold"],
            "safety_ucb_threshold": thresholds["safety_ucb_threshold"],
            "regret_ucb_threshold": thresholds["regret_ucb_threshold"],
        }
    )
