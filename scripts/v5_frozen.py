from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from concordia.feasibility import (
    RegimeDefinition,
    RobustShiftDetector,
    V5ModelBundle,
    V5_FEATURE_SCHEMA,
    expand_v5_features,
)
from concordia.selective import RegimeConditionedPolicy


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifacts/v5/freeze_manifest.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def verify_frozen() -> dict:
    if not MANIFEST.is_file():
        raise RuntimeError("v5 final evaluation is forbidden before freeze")
    manifest = json.loads(MANIFEST.read_text())
    for relative, expected in manifest["frozen_config_hashes"].items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen v5 config changed: {relative}")
    for relative, expected in manifest["deployment_code_hashes"].items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"post-freeze deployment code change detected: {relative}")
    for relative, expected in manifest["artifact_hashes"].items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen v5 artifact changed: {relative}")
    return manifest


def load_deployment():
    verify_frozen()
    regime_package = json.loads(
        (ROOT / "artifacts/studies/v5_shift_detection/regime_definition.json").read_text()
    )
    shift_package = json.loads(
        (ROOT / "artifacts/studies/v5_shift_detection/shift_detector.json").read_text()
    )
    analytical = json.loads(
        (ROOT / "artifacts/studies/v5_shift_detection/calibrated_model.json").read_text()
    )
    correction = json.loads(
        (ROOT / "artifacts/studies/v5_micro_calibration/micro_correction_package.json").read_text()
    )
    safety = json.loads(
        (ROOT / "artifacts/studies/v5_micro_calibration/micro_safety_package.json").read_text()
    )
    thresholds = yaml.safe_load(
        (ROOT / "configs/v5/frozen_thresholds.yaml").read_text()
    )["thresholds"]
    return (
        RegimeDefinition.from_dict(regime_package["definition"]),
        RobustShiftDetector.from_dict(shift_package["detector"]),
        shift_package["feature_names"],
        V5ModelBundle.from_packages(analytical, correction, safety),
        thresholds,
    )


def prepare_cases(cases: list[dict], regime, shift, shift_names, bundle):
    enriched = []
    for case in cases:
        features = expand_v5_features(case)
        shift_matrix = np.asarray([[features[name] for name in shift_names]], dtype=float)
        dss = float(shift.score(shift_matrix)[0])
        shift_class = shift.classify(shift_matrix)[0]
        features["dss_penetration_interaction"] = (
            dss * features["navigation_penetration"]
        )
        enriched.append(
            {
                **case,
                "features": features,
                "regime": regime.route(features),
                "shift_class": shift_class,
                "domain_shift_score": dss,
            }
        )
    matrix = np.asarray(
        [[row["features"][name] for name in V5_FEATURE_SCHEMA] for row in enriched],
        dtype=float,
    )
    predictions = bundle.predict(matrix, [row["regime"] for row in enriched])
    return enriched, predictions


def analytical_policy(thresholds: dict, *, variant: str = "V5-RD"):
    table = thresholds["probability_thresholds"]
    regimes = sorted(table)
    if variant == "V5-G":
        table = {
            regime: {
                shift: thresholds["global_probability_threshold"]
                for shift in ("IN_DISTRIBUTION", "MILD_SHIFT", "STRONG_SHIFT")
            }
            for regime in regimes
        }
    elif variant == "V5-R":
        table = {
            regime: {
                shift: table[regime]["IN_DISTRIBUTION"]
                for shift in ("IN_DISTRIBUTION", "MILD_SHIFT", "STRONG_SHIFT")
            }
            for regime in regimes
        }
    return RegimeConditionedPolicy(
        table,
        float(thresholds["shift_probability_penalty"]),
        float(thresholds["micro_success_threshold"]),
        float(thresholds["micro_safety_threshold"]),
        use_shift_gate=variant not in {"V5-G", "V5-R"},
        use_micro_correction=variant == "V5-F",
        use_micro_safety_veto=variant in {"V5-RS", "V5-F"},
    )


def microscopic_policy(thresholds: dict):
    return RegimeConditionedPolicy(
        thresholds["microscopic_probability_thresholds"],
        float(thresholds["microscopic_shift_probability_penalty"]),
        float(thresholds["micro_success_threshold"]),
        float(thresholds["micro_safety_threshold"]),
        use_shift_gate=True,
        use_micro_correction=True,
        use_micro_safety_veto=True,
    )
