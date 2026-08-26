from __future__ import annotations

import json

import yaml

from concordia.feasibility.calibration_v4 import ProbabilityCalibrator
from concordia.safety_v8.classifier import SafetyClassifier
from concordia.selective_v8.policy import SafetyFilteredUpliftPolicy
from concordia.selective_v8.safety_filter import CalibratedSafetyFilter
from concordia.selective_v8.traffic_ranker import TrafficRanker
from concordia.uplift_v7.learners import RegressionModel
from v8_common import ROOT, payload_hash, sha256


MANIFEST = ROOT / "artifacts/v8/freeze_manifest.json"
FROZEN = (
    "configs/v8/frozen_traffic_ranker.yaml",
    "configs/v8/frozen_safety_classifier.yaml",
    "configs/v8/frozen_safety_calibration.yaml",
    "configs/v8/frozen_regret_model.yaml",
    "configs/v8/frozen_policy.yaml",
    "configs/v8/frozen_thresholds.yaml",
)


def verify_frozen() -> dict:
    if not MANIFEST.is_file():
        raise RuntimeError("v8 final evaluation is forbidden before freeze")
    manifest = json.loads(MANIFEST.read_text())
    if payload_hash(manifest) != manifest.get("manifest_self_hash"):
        raise RuntimeError("v8 freeze manifest changed")
    for section in ("frozen_config_hashes", "development_artifact_hashes", "deployment_code_hashes"):
        for relative, expected in manifest[section].items():
            path = ROOT / relative
            if not path.is_file() or sha256(path) != expected:
                raise RuntimeError(f"post-freeze v8 change detected: {relative}")
    return manifest


def load_policy() -> SafetyFilteredUpliftPolicy:
    verify_frozen()
    values = [yaml.safe_load((ROOT / relative).read_text()) for relative in FROZEN]
    traffic, safety, calibration, regret, policy, thresholds = values
    return SafetyFilteredUpliftPolicy(
        TrafficRanker.from_dict(traffic["traffic_ranker"]),
        CalibratedSafetyFilter(
            SafetyClassifier.from_dict(safety["classifier"]),
            ProbabilityCalibrator.from_dict(calibration["calibrator"]),
            float(thresholds["unsafe_probability_threshold"]),
        ),
        RegressionModel.from_dict(regret["regret_model"]),
        float(thresholds["traffic_rank_percentile_cutoff"]),
        float(thresholds["regret_threshold"]),
        str(policy["policy_name"]),
    )
