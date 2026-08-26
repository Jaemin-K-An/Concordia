#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from concordia.feasibility.calibration_v4 import calibration_error
from concordia.safety_v8.evaluation import classification_metrics
from concordia.v9.evaluation import within_state_ranking_metrics
from concordia.v9.pairwise import PairwiseActionRanker
from concordia.v9.safety import ActionSafetyModel, unsafe_label
from concordia.v9.surrogate import CandidateScreen, StateActionTrafficModel


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "artifacts/studies/v9_actionability/raw_metrics.json"
MODEL_DIR = ROOT / "artifacts/studies/v9_action_models"
ROLLOUT_DIR = ROOT / "artifacts/studies/v9_rollout_validation"
POLICY_DIR = ROOT / "artifacts/studies/v9_policy_validation"
HEURISTICS = (
    "expected_accepted_user_count",
    "expected_rerouted_flow",
    "destination_capacity_slack",
    "route_entropy",
    "action_concentration_index",
    "expected_acceptance_probability",
    "reroute_fraction",
)


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _role(rows: list[dict], role: str, *, exhaustive: bool = False) -> list[dict]:
    return [
        row for row in rows
        if row["development_role"] == role
        and row["action_id"] != "B6_ALWAYS_ON_REFERENCE"
        and (not exhaustive or row["exhaustive_oracle"])
    ]


def _oracle_sets(rows: list[dict]) -> tuple[dict[str, list[int]], dict[str, set[int]]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["state_id"]].append(index)
    oracle = {}
    for state_id, indices in groups.items():
        values = np.asarray([
            float(rows[index]["outcomes"]["tau_t_relative"])
            if (
                float(rows[index]["outcomes"]["tau_s"]) <= 0.25
                and float(rows[index]["outcomes"]["max_regret"]) <= 0.08
                and bool(rows[index]["outcomes"]["legal"])
            )
            else -1e9
            for index in indices
        ])
        maximum = float(values.max())
        oracle[state_id] = {
            indices[offset]
            for offset in np.flatnonzero(np.isclose(values, maximum, atol=1e-12))
        }
    return groups, oracle


def _topm_recall(
    scores: np.ndarray, groups: dict[str, list[int]], oracle: dict[str, set[int]], top_m: int
) -> float:
    hits = 0
    for state_id, indices in groups.items():
        order = sorted(indices, key=lambda index: (-float(scores[index]), index))[:top_m]
        hits += int(bool(oracle[state_id].intersection(order)))
    return hits / len(groups)


def _select_ensemble_weights(components: np.ndarray, rows: list[dict]) -> tuple[float, ...]:
    groups, oracle = _oracle_sets(rows)
    rng = np.random.default_rng(20260902)
    best_value = (-1.0, -1.0, -1e9)
    best_weights = None
    candidate_weights = [np.eye(len(components))[index] for index in range(len(components))]
    candidate_weights.extend(rng.normal(size=(50_000, len(components))))
    for weights in candidate_weights:
        scores = weights @ components
        objective = (
            _topm_recall(scores, groups, oracle, 5),
            _topm_recall(scores, groups, oracle, 7),
            -float(np.linalg.norm(weights)),
        )
        if objective > best_value:
            best_value = objective
            best_weights = weights.copy()
    if best_weights is None:
        raise RuntimeError("v9 ensemble search produced no candidate")
    return tuple(map(float, best_weights))


def _safety_development(train: list[dict], calibration: list[dict], validation: list[dict]):
    reports = []
    fitted = []
    y_cal = np.asarray([unsafe_label(row) for row in calibration], dtype=int)
    y_val = np.asarray([unsafe_label(row) for row in validation], dtype=int)
    for index, (model_id, kind) in enumerate((
        ("S0_logistic_interactions", "logistic"),
        ("S1_random_forest", "random_forest"),
        ("S2_gradient_boosting", "gradient_boosting"),
    )):
        base = ActionSafetyModel.build(model_id, kind, 9400 + index, 3.0).fit(train)
        calibrations = []
        for method in ("raw", "platt", "beta", "isotonic"):
            candidate = ActionSafetyModel.from_dict({
                "classifier": base.classifier.to_dict(), "calibrator": None,
            }).calibrate(calibration, method)
            probability = candidate.predict_proba(calibration)
            diagnostics = calibration_error(y_cal, probability)
            calibrations.append((diagnostics["brier_score"] + diagnostics["ece"], candidate, diagnostics))
        _objective, selected, calibration_diagnostics = min(
            calibrations, key=lambda value: (value[0], value[1].calibrator.method)
        )
        validation_probability = selected.predict_proba(validation)
        report = {
            "model_id": model_id,
            "kind": kind,
            "selected_calibration": selected.calibrator.method,
            "calibration_diagnostics": calibration_diagnostics,
            "calibration_classification_at_0_5": classification_metrics(
                y_cal, selected.predict_proba(calibration), 0.5
            ),
            "validation_classification_at_0_5": classification_metrics(
                y_val, validation_probability, 0.5
            ),
        }
        reports.append(report)
        fitted.append(selected)
    selected_index = max(
        range(len(reports)),
        key=lambda index: (
            reports[index]["calibration_classification_at_0_5"]["pr_auc_average_precision"],
            -reports[index]["calibration_diagnostics"]["brier_score"],
        ),
    )
    _write(MODEL_DIR / "safety_model_comparison.json", {
        "selection_partition": "calibration",
        "models": reports,
        "selected_model_id": reports[selected_index]["model_id"],
    })
    _write(MODEL_DIR / "selected_safety_model.json", fitted[selected_index].to_dict())
    return reports[selected_index]


def run() -> Path:
    if (ROOT / "artifacts/studies/v9_micro_holdout/summary.json").exists():
        raise RuntimeError("v9 validation repair cannot inspect a materialized final holdout")
    rows = json.loads(DATASET.read_text())
    train = _role(rows, "train")
    calibration = _role(rows, "calibration", exhaustive=True)
    validation = _role(rows, "validation", exhaustive=True)
    traffic_models = tuple(
        StateActionTrafficModel.from_dict(json.loads((MODEL_DIR / f"{name}.json").read_text()))
        for name in ("T0_random_forest", "T1_gradient_boosting", "T2_hist_gradient_boosting")
    )
    pairwise = PairwiseActionRanker(iterations=600).fit(train)
    provisional = CandidateScreen(traffic_models, pairwise, HEURISTICS, tuple([1.0] * 11))
    components = provisional.component_scores(calibration)
    weights = _select_ensemble_weights(components, calibration)
    screen = CandidateScreen(traffic_models, pairwise, HEURISTICS, weights)
    calibration_scores = screen.predict(calibration)
    validation_scores = screen.predict(validation)
    calibration_metrics = within_state_ranking_metrics(calibration, calibration_scores, 5)
    validation_top5 = within_state_ranking_metrics(validation, validation_scores, 5)
    validation_top7 = within_state_ranking_metrics(validation, validation_scores, 7)
    pairwise_validation = within_state_ranking_metrics(
        validation, pairwise.predict(validation), 5
    )
    _write(MODEL_DIR / "pairwise_ranker.json", pairwise.to_dict())
    _write(MODEL_DIR / "candidate_screen.json", screen.to_dict())
    safety_report = _safety_development(
        _role(rows, "train"), _role(rows, "calibration"), _role(rows, "validation")
    )
    gate_b_pass = validation_top5["top_5_oracle_recall"] >= 0.80
    ranking = {
        "repair_round": 2,
        "registered_changes": [
            "state_action_interactions",
            "pairwise_within_state_ranker",
            "calibration_selected_candidate_ensemble",
            "top_7_development_only_comparison",
        ],
        "calibration": calibration_metrics,
        "validation_top_5": validation_top5,
        "validation_top_7_diagnostic": validation_top7,
        "pairwise_only_validation": pairwise_validation,
        "gate_B_threshold": 0.80,
        "gate_B_pass": gate_b_pass,
        "selected_safety_model_validation": safety_report,
    }
    _write(MODEL_DIR / "ranking_repair_validation.json", ranking)
    _write(ROLLOUT_DIR / "summary.json", {
        "gate_C_threshold": 0.70,
        "status": "not_reached" if not gate_b_pass else "pending",
        "reason": "Gate B failed after registered ranking repair" if not gate_b_pass else None,
        "evaluation_rollout_seed_overlap": 0,
        "actual_rollout_count": 0,
        "final_holdout_materialized": False,
    })
    gate_report = {
        "development_repair_rounds_used": 1,
        "round_1_action_space": "not_triggered_gate_A_passed",
        "round_2_ranking": "applied_and_exhausted",
        "round_3_rollout_safety": "not_eligible_gate_B_failed",
        "gate_A": {"value": 0.458, "threshold": 0.40, "passed": True},
        "gate_B": {
            "value": validation_top5["top_5_oracle_recall"],
            "threshold": 0.80,
            "passed": gate_b_pass,
        },
        "gate_C": {"value": None, "threshold": 0.70, "passed": False, "status": "not_reached"},
        "gate_D": {"value": None, "passed": False, "status": "not_reached"},
        "policy_freeze_authorized": False,
        "final_holdout_authorized": False,
        "final_model_selection": "none_safe_gate_preserving_stop",
        "rl_used": False,
    }
    _write(POLICY_DIR / "gate_validation.json", gate_report)
    _write(POLICY_DIR / "repair_ledger.json", {
        "maximum_registered_rounds": 3,
        "rounds_consumed": 1,
        "changes": ranking["registered_changes"],
        "forbidden_changes_made": [],
        "minimum_benefit_unchanged": 0.005,
        "safety_delta_unchanged": 0.25,
        "maximum_regret_unchanged": 0.08,
        "final_seeds_inspected": False,
    })
    print(POLICY_DIR / "gate_validation.json")
    return POLICY_DIR / "gate_validation.json"


if __name__ == "__main__":
    run()
