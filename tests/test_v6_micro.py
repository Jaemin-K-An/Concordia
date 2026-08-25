from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from concordia.errors import ValidationError
from concordia.feasibility.calibration_v4 import ProbabilityCalibrator
from concordia.micro_v6 import (
    MICRO_V6_FEATURE_SCHEMA,
    MicroSuccessPredictor,
    V6Policy,
    build_safe_micro_label,
    claim_allowed,
    validate_predecision_features,
)


ROOT = Path(__file__).resolve().parents[1]


def _run(ttt: float, risk: float, regret: float = 0.0, legal: bool = True) -> dict:
    return {
        "total_travel_time_seconds": ttt,
        "safety": {"cvar_drac_95": risk},
        "maximum_affected_regret": regret,
        "all_executed_routes_legal": legal,
    }


def test_safe_micro_success_requires_all_four_constraints() -> None:
    label = build_safe_micro_label(
        _run(1000.0, 1.0),
        _run(980.0, 1.2, 0.08, True),
        minimum_relative_ttt_gain=0.01,
        safety_margin=0.25,
        regret_limit=0.08,
    )
    assert label.safe_micro_success
    assert label.diagnostic_class == "S1_safe_beneficial"
    unsafe = build_safe_micro_label(
        _run(1000.0, 1.0),
        _run(970.0, 1.3, 0.01, True),
        minimum_relative_ttt_gain=0.01,
        safety_margin=0.25,
        regret_limit=0.08,
    )
    assert not unsafe.safe_micro_success
    assert unsafe.diagnostic_class == "U1_unsafe_beneficial"


def test_predecision_feature_window_cannot_cross_decision() -> None:
    features = {name: 0.0 for name in MICRO_V6_FEATURE_SCHEMA}
    with pytest.raises(ValidationError):
        validate_predecision_features(
            {
                "decision_time": 30.0,
                "feature_observation_end_time": 31.0,
                "features_pre_decision": features,
            }
        )


def test_v6_final_micro_seeds_are_disjoint() -> None:
    config = yaml.safe_load((ROOT / "configs/v6/micro_design.yaml").read_text())
    assert not set(config["development_seeds"]) & set(config["final_holdout_seeds"])
    role_seeds = {
        seed for values in config["development_roles"].values() for seed in values
    }
    assert role_seeds == set(config["development_seeds"])


def test_v6_pilot_pairing_contract() -> None:
    checkpoint = ROOT / "artifacts/studies/v6_micro_dataset/development_checkpoint.json"
    completed = ROOT / "artifacts/studies/v6_micro_dataset/raw_metrics.json"
    source = checkpoint if checkpoint.is_file() else completed
    if not source.is_file():
        pytest.skip("v6 microscopic dataset not generated")
    rows = json.loads(source.read_text())
    assert rows
    assert all(row["pairing"]["same_seed"] for row in rows)
    assert all(row["pairing"]["same_network_hash"] for row in rows)
    assert all(row["pairing"]["same_route_file_hash"] for row in rows)
    for row in rows:
        validate_predecision_features(row)


class _ConstantModel:
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, matrix) -> np.ndarray:
        return np.full(len(matrix), self.probability, dtype=float)


def _policy(composite: float, unsafe: float, architecture: str = "C_composite_plus_safety_veto"):
    names = tuple(MICRO_V6_FEATURE_SCHEMA)
    predictor = MicroSuccessPredictor(
        "constant", "global", names, _ConstantModel(composite)
    )
    raw = ProbabilityCalibrator("raw", {})
    return V6Policy(
        predictor,
        raw,
        _ConstantModel(composite),
        raw,
        _ConstantModel(unsafe),
        raw,
        architecture,
        0.80,
        0.10,
        0.0,
    )


def _feature_row() -> dict:
    return {
        "case_id": "policy-contract",
        "features_pre_decision": {name: 0.0 for name in MICRO_V6_FEATURE_SCHEMA},
    }


def test_v6_abstention_executes_unchanged_b1() -> None:
    decision = _policy(0.10, 0.0, "A_composite").decide([_feature_row()])[0]
    assert not decision["intervene"]
    assert decision["executed_policy"] == "B1"
    assert decision["reason"] == "micro_abstain"


def test_v6_unsafe_prediction_is_never_executed() -> None:
    decision = _policy(0.95, 0.90).decide([_feature_row()])[0]
    assert not decision["intervene"]
    assert decision["executed_policy"] == "B1"
    assert decision["reason"] == "safety_veto"


def test_v6_failed_metrics_forbid_success_claim() -> None:
    metrics = {"intervention_count": 50, "precision": 0.79, "safety_violation_count": 0}
    assert not claim_allowed(metrics, minimum_interventions=30, required_precision=0.80)
    metrics.update({"precision": 0.90, "safety_violation_count": 1})
    assert not claim_allowed(metrics, minimum_interventions=30, required_precision=0.80)


def test_v6_predictor_inference_contract() -> None:
    policy = _policy(0.95, 0.01)
    durations = []
    for _ in range(30):
        started = time.perf_counter()
        policy.decide([_feature_row()])
        durations.append(time.perf_counter() - started)
    assert float(np.quantile(durations, 0.95)) < 0.1


def test_v6_freeze_manifest_payload_hash_detects_change() -> None:
    from scripts.v6_frozen import payload_hash

    manifest = {"complete": True, "source_commit": "abc"}
    digest = payload_hash(manifest)
    manifest["source_commit"] = "def"
    assert payload_hash(manifest) != digest
