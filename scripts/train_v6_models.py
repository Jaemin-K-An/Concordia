#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from concordia.feasibility.calibration_v4 import ProbabilityCalibrator, calibration_error
from concordia.micro_v6.features import MICRO_V6_FEATURE_GROUPS, MICRO_V6_FEATURE_SCHEMA
from concordia.micro_v6.modeling import (
    average_precision,
    binary_auc,
    calibration_slope_intercept,
    feature_matrix,
    fit_candidate,
    row_regimes,
    V6TreeModel,
)
from concordia.micro_v6.policy import V6Policy, selected_mask, selective_metrics


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "artifacts/studies/v6_micro_dataset/raw_metrics.json"
MODEL_DIR = ROOT / "artifacts/studies/v6_micro_model_selection"
CALIBRATION_DIR = ROOT / "artifacts/studies/v6_micro_calibration"
POLICY_DIR = ROOT / "artifacts/studies/v6_policy_validation"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _labels(rows: Sequence[dict], key: str) -> np.ndarray:
    if key == "unsafe":
        return np.asarray([not bool(row["label"]["safety_pass"]) for row in rows], dtype=int)
    return np.asarray([bool(row["label"][key]) for row in rows], dtype=int)


def _calibration_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    metrics = calibration_error(labels.tolist(), probabilities.tolist())
    slope, intercept = calibration_slope_intercept(labels, probabilities)
    high = probabilities >= 0.70
    metrics.update(
        {
            "slope": slope,
            "intercept": intercept,
            "high_confidence_count": int(high.sum()),
            "high_confidence_observed_success": (
                float(labels[high].mean()) if high.any() else None
            ),
            "auc": binary_auc(labels, probabilities),
            "average_precision": average_precision(labels, probabilities),
        }
    )
    return metrics


def _fit_calibrator(
    raw_probability: np.ndarray,
    labels: np.ndarray,
    methods: Sequence[str],
) -> tuple[ProbabilityCalibrator, list[dict]]:
    comparison = []
    candidates = []
    for method in ("raw", *methods):
        calibrator = ProbabilityCalibrator(method).fit(raw_probability, labels)
        probability = calibrator.predict(raw_probability)
        metrics = _calibration_metrics(labels, probability)
        comparison.append({"method": method, "metrics": metrics})
        candidates.append((metrics["ece"], metrics["brier_score"], method, calibrator))
    _, _, _, selected = min(candidates, key=lambda value: value[:3])
    return selected, comparison


def _fit_auxiliary(
    train: list[dict],
    calibration: list[dict],
    feature_names: Sequence[str],
    methods: Sequence[str],
    seed: int,
) -> tuple[Any, ProbabilityCalibrator, Any, ProbabilityCalibrator, dict]:
    train_matrix = feature_matrix(train, feature_names)
    calibration_matrix = feature_matrix(calibration, feature_names)
    benefit_labels = _labels(train, "traffic_benefit_pass")
    unsafe_labels = _labels(train, "unsafe")
    benefit = V6TreeModel(
        "micro_traffic_benefit", "gradient_boosting", tuple(feature_names), seed=seed,
        tree_count=100, max_depth=2, minimum_leaf=8, learning_rate=0.06,
    ).fit(train_matrix, benefit_labels)
    safety = V6TreeModel(
        "micro_safety_violation", "gradient_boosting", tuple(feature_names), seed=seed + 1,
        tree_count=100, max_depth=2, minimum_leaf=8, learning_rate=0.06,
    ).fit(train_matrix, unsafe_labels)
    benefit_cal, benefit_comparison = _fit_calibrator(
        benefit.predict_proba(calibration_matrix),
        _labels(calibration, "traffic_benefit_pass"),
        methods,
    )
    safety_cal, safety_comparison = _fit_calibrator(
        safety.predict_proba(calibration_matrix), _labels(calibration, "unsafe"), methods
    )
    return benefit, benefit_cal, safety, safety_cal, {
        "benefit": benefit_comparison,
        "safety": safety_comparison,
    }


def _stage1_threshold(validation: list[dict], thresholds: Sequence[float]) -> tuple[float, list[dict]]:
    labels = _labels(validation, "safe_micro_success").astype(bool)
    analytical = np.asarray(
        [row["features_pre_decision"]["analytical_success_probability"] for row in validation]
    )
    frontier = []
    for threshold in thresholds:
        selected = analytical >= float(threshold)
        recall = float((selected & labels).sum() / labels.sum()) if labels.any() else 0.0
        frontier.append(
            {
                "threshold": float(threshold),
                "screen_recall": recall,
                "screen_coverage": float(selected.mean()),
            }
        )
    feasible = [row for row in frontier if row["screen_recall"] >= 0.95]
    selected = max(feasible, key=lambda row: row["threshold"]) if feasible else frontier[0]
    return float(selected["threshold"]), frontier


def _policy_candidate(
    predictor,
    calibrator,
    benefit,
    benefit_cal,
    safety,
    safety_cal,
    architecture: str,
    success_threshold: float,
    safety_threshold: float,
    stage1_threshold: float,
    conformal: bool = False,
) -> V6Policy:
    return V6Policy(
        predictor,
        calibrator,
        benefit,
        benefit_cal,
        safety,
        safety_cal,
        architecture,
        success_threshold,
        safety_threshold,
        stage1_threshold,
        conformal,
    )


def _conformal_thresholds(
    calibration_rows: list[dict], probabilities: np.ndarray, failure_levels: Sequence[float]
) -> list[tuple[float, float]]:
    labels = _labels(calibration_rows, "safe_micro_success").astype(bool)
    negative = probabilities[~labels]
    if not len(negative):
        return []
    values = []
    for alpha in failure_levels:
        quantile = min(1.0, max(0.0, 1.0 - float(alpha)))
        try:
            threshold = float(np.quantile(negative, quantile, method="higher"))
        except TypeError:
            threshold = float(np.quantile(negative, quantile, interpolation="higher"))
        values.append((float(alpha), threshold))
    return values


def _evaluate_policy(policy: V6Policy, rows: list[dict]) -> tuple[dict, list[dict]]:
    decisions = policy.decide(rows)
    return selective_metrics(rows, selected_mask(decisions)), decisions


def run(*, force: bool = False) -> dict:
    existing_summary = POLICY_DIR / "validation_summary.json"
    existing_package = POLICY_DIR / "selected_policy_package.json"
    if existing_summary.is_file() and existing_package.is_file() and not force:
        summary = json.loads(existing_summary.read_text())
        print(json.dumps(summary, indent=2, sort_keys=True))
        return summary
    if not DATASET.is_file():
        raise RuntimeError("v6 microscopic development dataset is missing")
    rows = json.loads(DATASET.read_text())
    if len(rows) < 500:
        raise RuntimeError("v6 model training requires at least 500 paired SUMO cases")
    roles = {
        role: [row for row in rows if row["development_role"] == role]
        for role in ("train", "calibration", "validation")
    }
    config = yaml.safe_load((ROOT / "configs/v6/model.yaml").read_text())
    train = roles["train"]
    calibration = roles["calibration"]
    validation = roles["validation"]
    train_matrix = feature_matrix(train)
    calibration_matrix = feature_matrix(calibration)
    validation_matrix = feature_matrix(validation)
    train_labels = _labels(train, "safe_micro_success")
    calibration_labels = _labels(calibration, "safe_micro_success")
    validation_labels = _labels(validation, "safe_micro_success")
    train_regimes = row_regimes(train)
    calibration_regimes = row_regimes(calibration)
    validation_regimes = row_regimes(validation)
    methods = list(config["calibration_methods"])

    benefit, benefit_cal, safety, safety_cal, auxiliary_comparison = _fit_auxiliary(
        train, calibration, MICRO_V6_FEATURE_SCHEMA, methods, int(config["seed"]) + 100
    )
    candidates = []
    fitted = {}
    for offset, name in enumerate(config["candidate_models"]):
        predictor = fit_candidate(
            name,
            train_matrix,
            train_labels,
            train_regimes,
            seed=int(config["seed"]) + offset,
            minimum_regime_size=int(config["minimum_regime_size"]),
        )
        raw_calibration = predictor.predict_proba(calibration_matrix, calibration_regimes)
        calibrator, calibration_comparison = _fit_calibrator(
            raw_calibration, calibration_labels, methods
        )
        validation_probability = calibrator.predict(
            predictor.predict_proba(validation_matrix, validation_regimes)
        )
        entry = {
            "name": name,
            "strategy": predictor.strategy,
            "selected_calibration": calibrator.method,
            "calibration_comparison": calibration_comparison,
            "validation_metrics": _calibration_metrics(validation_labels, validation_probability),
            "importance": predictor.importance(),
            "regime_models": sorted((predictor.regime_models or {}).keys()),
        }
        candidates.append(entry)
        fitted[name] = (predictor, calibrator)

    stage1_threshold, stage1_frontier = _stage1_threshold(
        validation, config["stage1_probability_thresholds"]
    )
    frontier = []
    policy_objects = {}
    for name, (predictor, calibrator) in fitted.items():
        validation_composite = calibrator.predict(
            predictor.predict_proba(validation_matrix, validation_regimes)
        )
        validation_benefit = benefit_cal.predict(benefit.predict_proba(validation_matrix))
        for architecture in config["architectures"]:
            safety_thresholds = (
                [1.0]
                if architecture == "A_composite"
                else config["safety_probability_thresholds"]
            )
            architecture_probability = (
                validation_benefit
                if architecture == "B_benefit_plus_safety_veto"
                else validation_composite
            )
            success_thresholds = sorted(
                set(float(value) for value in config["probability_thresholds"])
                | set(float(value) for value in architecture_probability)
            )
            for success_threshold in success_thresholds:
                for safety_threshold in safety_thresholds:
                    policy = _policy_candidate(
                        predictor,
                        calibrator,
                        benefit,
                        benefit_cal,
                        safety,
                        safety_cal,
                        architecture,
                        float(success_threshold),
                        float(safety_threshold),
                        stage1_threshold,
                    )
                    metrics, _ = _evaluate_policy(policy, validation)
                    key = (
                        f"{name}|{architecture}|p={success_threshold:.12f}|"
                        f"s={float(safety_threshold):.2f}|classical"
                    )
                    feasible = bool(
                        metrics["intervention_count"] > 0
                        and metrics["precision"] >= 0.80
                        and metrics["safety_violation_count"] == 0
                    )
                    frontier.append(
                        {
                            "key": key,
                            "model": name,
                            "architecture": architecture,
                            "success_threshold": float(success_threshold),
                            "safety_threshold": float(safety_threshold),
                            "method": "classical",
                            "feasible": feasible,
                            "metrics": metrics,
                        }
                    )
                    policy_objects[key] = policy

        calibration_probability = calibrator.predict(
            predictor.predict_proba(calibration_matrix, calibration_regimes)
        )
        for alpha, success_threshold in _conformal_thresholds(
            calibration, calibration_probability, config["conformal_failure_levels"]
        ):
            for safety_threshold in config["safety_probability_thresholds"]:
                policy = _policy_candidate(
                    predictor,
                    calibrator,
                    benefit,
                    benefit_cal,
                    safety,
                    safety_cal,
                    "C_composite_plus_safety_veto",
                    success_threshold,
                    float(safety_threshold),
                    stage1_threshold,
                    True,
                )
                metrics, _ = _evaluate_policy(policy, validation)
                key = (
                    f"{name}|V6-C|alpha={alpha:.2f}|p={success_threshold:.6f}|"
                    f"s={float(safety_threshold):.2f}"
                )
                feasible = bool(
                    metrics["intervention_count"] > 0
                    and metrics["precision"] >= 0.80
                    and metrics["safety_violation_count"] == 0
                )
                frontier.append(
                    {
                        "key": key,
                        "model": name,
                        "architecture": "C_composite_plus_safety_veto",
                        "success_threshold": success_threshold,
                        "safety_threshold": float(safety_threshold),
                        "method": "split_conformal_negative_score",
                        "failure_level": alpha,
                        "feasible": feasible,
                        "metrics": metrics,
                    }
                )
                policy_objects[key] = policy

    feasible = [row for row in frontier if row["feasible"]]
    if feasible:
        selected_row = max(
            feasible,
            key=lambda row: (
                row["metrics"]["opportunity_recovery_rate"],
                row["metrics"]["coverage"],
                row["metrics"]["precision"],
                -candidates[[item["name"] for item in candidates].index(row["model"])][
                    "validation_metrics"
                ]["ece"],
                row["architecture"] == "A_composite",
            ),
        )
    else:
        diagnostic_best = max(
            frontier,
            key=lambda row: (
                row["metrics"]["precision"],
                -row["metrics"]["safety_violation_count"],
                row["metrics"]["opportunity_recovery_rate"],
            ),
        )
        fallback_model = max(
            candidates,
            key=lambda row: (
                row["validation_metrics"]["average_precision"],
                row["validation_metrics"]["auc"],
                -row["validation_metrics"]["ece"],
            ),
        )
        predictor, calibrator = fitted[fallback_model["name"]]
        fallback_policy = _policy_candidate(
            predictor,
            calibrator,
            benefit,
            benefit_cal,
            safety,
            safety_cal,
            "C_composite_plus_safety_veto",
            1.000001,
            min(float(value) for value in config["safety_probability_thresholds"]),
            stage1_threshold,
        )
        selected_row = {
            "key": f"{fallback_model['name']}|constraint_infeasible|safe_abstention",
            "model": fallback_model["name"],
            "architecture": "C_composite_plus_safety_veto",
            "success_threshold": 1.000001,
            "safety_threshold": min(
                float(value) for value in config["safety_probability_thresholds"]
            ),
            "method": "safe_abstention_when_validation_constraints_infeasible",
            "feasible": False,
            "validation_frontier_best_diagnostic": diagnostic_best,
            "metrics": selective_metrics(validation, [False] * len(validation)),
        }
        policy_objects[selected_row["key"]] = fallback_policy
    selected_policy = policy_objects[selected_row["key"]]
    selected_metrics, selected_decisions = _evaluate_policy(selected_policy, validation)

    ablations = []
    for group in (
        "traffic_temporal",
        "analytical",
        "topology",
        "preference",
        "penetration",
        "safety",
    ):
        omitted = set(MICRO_V6_FEATURE_GROUPS[group])
        names = tuple(name for name in MICRO_V6_FEATURE_SCHEMA if name not in omitted)
        ablation_train = feature_matrix(train, names)
        ablation_calibration = feature_matrix(calibration, names)
        predictor = fit_candidate(
            selected_row["model"],
            ablation_train,
            train_labels,
            train_regimes,
            names,
            seed=int(config["seed"]) + 300 + len(ablations),
            minimum_regime_size=int(config["minimum_regime_size"]),
        )
        calibrator, _ = _fit_calibrator(
            predictor.predict_proba(ablation_calibration, calibration_regimes),
            calibration_labels,
            methods,
        )
        aux = _fit_auxiliary(
            train, calibration, names, methods, int(config["seed"]) + 400 + len(ablations)
        )
        policy = _policy_candidate(
            predictor,
            calibrator,
            aux[0],
            aux[1],
            aux[2],
            aux[3],
            selected_policy.architecture,
            selected_policy.success_threshold,
            selected_policy.safety_threshold,
            selected_policy.stage1_threshold,
            selected_policy.conformal,
        )
        metrics, _ = _evaluate_policy(policy, validation)
        probability = calibrator.predict(
            predictor.predict_proba(feature_matrix(validation, names), validation_regimes)
        )
        ablations.append(
            {
                "ablation": f"without_{group}",
                "feature_count": len(names),
                "metrics": metrics,
                "auc": binary_auc(validation_labels, probability),
                "average_precision": average_precision(validation_labels, probability),
            }
        )
    no_stage1 = V6Policy.from_dict(selected_policy.to_dict())
    no_stage1.stage1_threshold = 0.0
    no_stage1_metrics, _ = _evaluate_policy(no_stage1, validation)
    no_veto = V6Policy.from_dict(selected_policy.to_dict())
    no_veto.architecture = "A_composite"
    no_veto.safety_threshold = 1.0
    no_veto_metrics, _ = _evaluate_policy(no_veto, validation)
    ablations.extend(
        [
            {"ablation": "no_stage1_screener", "metrics": no_stage1_metrics},
            {"ablation": "no_safety_veto", "metrics": no_veto_metrics},
        ]
    )

    penetration_analysis = []
    for penetration in sorted({row["condition"]["penetration"] for row in rows}):
        subset = [row for row in rows if row["condition"]["penetration"] == penetration]
        penetration_analysis.append(
            {
                "penetration": penetration,
                "sample_count": len(subset),
                "success_rate": float(_labels(subset, "safe_micro_success").mean()),
                "unsafe_rate": float(_labels(subset, "unsafe").mean()),
            }
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    POLICY_DIR.mkdir(parents=True, exist_ok=True)
    write_json(MODEL_DIR / "candidate_comparison.json", candidates)
    write_json(
        MODEL_DIR / "training_manifest.json",
        {
            "dataset_path": str(DATASET.relative_to(ROOT)),
            "development_only": True,
            "role_counts": {key: len(value) for key, value in roles.items()},
            "case_ids": {key: [row["case_id"] for row in value] for key, value in roles.items()},
            "feature_schema": list(MICRO_V6_FEATURE_SCHEMA),
            "final_holdout_case_ids": [],
            "final_holdout_used": False,
        },
    )
    write_json(MODEL_DIR / "penetration_analysis.json", penetration_analysis)
    write_json(
        CALIBRATION_DIR / "calibration_comparison.json",
        {"composite_candidates": candidates, "auxiliary": auxiliary_comparison},
    )
    write_json(POLICY_DIR / "precision_coverage_frontier.json", frontier)
    write_json(POLICY_DIR / "stage1_screening_frontier.json", stage1_frontier)
    write_json(POLICY_DIR / "ablation.json", ablations)
    write_json(POLICY_DIR / "selected_policy_package.json", selected_policy.to_dict())
    write_json(POLICY_DIR / "threshold_package.json", selected_row)
    write_json(POLICY_DIR / "validation_decisions.json", selected_decisions)
    validation_summary = {
        "selection_data": "development_validation_only",
        "selected": selected_row,
        "metrics": selected_metrics,
        "precision_constraint_met": selected_metrics["precision"] >= 0.80,
        "validation_safety_constraint_met": selected_metrics["safety_violation_count"] == 0,
        "stage1_threshold": stage1_threshold,
        "rl_used": False,
    }
    write_json(POLICY_DIR / "validation_summary.json", validation_summary)
    print(json.dumps(validation_summary, indent=2, sort_keys=True))
    return validation_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    run(force=arguments.force)
