#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import yaml

from concordia.safety_v8.features import ACTION_AWARE_FEATURE_SCHEMA
from concordia.selective_v8.policy import SafetyFilteredUpliftPolicy
from v7_frozen import verify_frozen as verify_v7
from v8_common import ROOT, payload_hash, sha256, write_json


FROZEN = {
    "traffic": ROOT / "configs/v8/frozen_traffic_ranker.yaml",
    "safety": ROOT / "configs/v8/frozen_safety_classifier.yaml",
    "calibration": ROOT / "configs/v8/frozen_safety_calibration.yaml",
    "regret": ROOT / "configs/v8/frozen_regret_model.yaml",
    "policy": ROOT / "configs/v8/frozen_policy.yaml",
    "thresholds": ROOT / "configs/v8/frozen_thresholds.yaml",
}
MANIFEST = ROOT / "artifacts/v8/freeze_manifest.json"


def _yaml(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False, width=120))


def run() -> Path:
    v7_manifest = verify_v7()
    selected_path = ROOT / "artifacts/studies/v8_policy_validation/selected_policy.json"
    summary_path = ROOT / "artifacts/studies/v8_policy_validation/summary.json"
    if not selected_path.is_file() or not summary_path.is_file():
        raise RuntimeError("v8 validation must finish before freeze")
    policy = SafetyFilteredUpliftPolicy.from_dict(json.loads(selected_path.read_text()))
    validation = json.loads(summary_path.read_text())
    _yaml(FROZEN["traffic"], {
        "study": "CONCORDIA v8",
        "component": "traffic_ranker",
        "traffic_ranker": policy.traffic_ranker.to_dict(),
    })
    _yaml(FROZEN["safety"], {
        "study": "CONCORDIA v8",
        "component": "action_aware_unsafe_classifier",
        "classifier": policy.safety_filter.classifier.to_dict(),
        "unsafe_label": "risk_adaptive > risk_b1 + 0.25",
    })
    _yaml(FROZEN["calibration"], {
        "study": "CONCORDIA v8",
        "component": "unsafe_probability_calibration",
        "calibrator": policy.safety_filter.calibrator.to_dict(),
    })
    _yaml(FROZEN["regret"], {
        "study": "CONCORDIA v8",
        "component": "retained_v7_regret_model",
        "regret_model": policy.regret_model.to_dict(),
        "v7_manifest_self_hash": v7_manifest["manifest_self_hash"],
    })
    _yaml(FROZEN["policy"], {
        "study": "CONCORDIA v8",
        "component": "rank_then_hard_safety_veto_policy",
        "policy_name": policy.policy_name,
        "decision_order": ["traffic_rank", "unsafe_probability_veto", "regret_veto", "legal_execution_gate"],
        "action_aware_feature_schema": list(ACTION_AWARE_FEATURE_SCHEMA),
        "legal_executable_predecision_required": True,
        "safe_abstention": bool(validation["safe_abstention"]),
    })
    _yaml(FROZEN["thresholds"], {
        "study": "CONCORDIA v8",
        "component": "deployment_thresholds",
        "traffic_rank_percentile_cutoff": policy.traffic_rank_percentile_cutoff,
        "unsafe_probability_threshold": policy.safety_filter.unsafe_probability_threshold,
        "regret_threshold": policy.regret_threshold,
        "safety_delta": 0.25,
    })
    artifacts = (
        "artifacts/studies/v8_safety_dataset/raw_metrics.json",
        "artifacts/studies/v8_safety_dataset/dataset_summary.json",
        "artifacts/studies/v8_safety_model_selection/model_comparison.json",
        "artifacts/studies/v8_safety_model_selection/state_vs_action.json",
        "artifacts/studies/v8_traffic_ranking/summary.json",
        "artifacts/studies/v8_policy_validation/frontier.json",
        "artifacts/studies/v8_policy_validation/summary.json",
        "artifacts/studies/v8_policy_validation/selected_policy.json",
    )
    code = (
        "src/concordia/safety_v8/labels.py",
        "src/concordia/safety_v8/features.py",
        "src/concordia/safety_v8/classifier.py",
        "src/concordia/safety_v8/calibration.py",
        "src/concordia/safety_v8/candidate_conditioned.py",
        "src/concordia/safety_v8/evaluation.py",
        "src/concordia/selective_v8/traffic_ranker.py",
        "src/concordia/selective_v8/safety_filter.py",
        "src/concordia/selective_v8/policy.py",
        "src/concordia/selective_v8/metrics.py",
        "scripts/v8_frozen.py",
    )
    manifest = {
        "study": "CONCORDIA v8 safety-filtered uplift ranking",
        "frozen_before_final_evaluation": True,
        "frozen_config_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in FROZEN.values()},
        "development_artifact_hashes": {relative: sha256(ROOT / relative) for relative in artifacts},
        "deployment_code_hashes": {relative: sha256(ROOT / relative) for relative in code},
        "feature_schema_hash": __import__("hashlib").sha256("\n".join(ACTION_AWARE_FEATURE_SCHEMA).encode()).hexdigest(),
        "unsafe_label_hash": __import__("hashlib").sha256(b"risk_adaptive > risk_b1 + 0.25").hexdigest(),
        "outcome_definition_hash": sha256(ROOT / "configs/v8/preregistration.yaml"),
        "split_definition_hash": sha256(ROOT / "configs/v8/safety_design.yaml"),
        "policy_definition_hash": sha256(FROZEN["policy"]),
        "v7_manifest_self_hash": v7_manifest["manifest_self_hash"],
        "final_holdout_evaluated": False,
        "rl_used": False,
    }
    manifest["manifest_self_hash"] = payload_hash(manifest)
    write_json(MANIFEST, manifest)
    print(MANIFEST)
    return MANIFEST


if __name__ == "__main__":
    run()
