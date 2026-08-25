from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from concordia.errors import ValidationError
from concordia.micro_v6 import (
    MICRO_V6_FEATURE_SCHEMA,
    build_safe_micro_label,
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
    if not checkpoint.is_file():
        pytest.skip("v6 pilot not generated")
    rows = json.loads(checkpoint.read_text())
    assert rows
    assert all(row["pairing"]["same_seed"] for row in rows)
    assert all(row["pairing"]["same_network_hash"] for row in rows)
    assert all(row["pairing"]["same_route_file_hash"] for row in rows)
    for row in rows:
        validate_predecision_features(row)
