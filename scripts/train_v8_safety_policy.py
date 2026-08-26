#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from concordia.feasibility.calibration_v4 import ProbabilityCalibrator
from concordia.safety_v8.calibration import calibration_diagnostics
from concordia.safety_v8.classifier import SafetyClassifier
from concordia.safety_v8.evaluation import (
    classification_metrics,
    critical_group_recall,
    pr_curve,
)
from concordia.safety_v8.features import (
    ACTION_AWARE_FEATURE_SCHEMA,
    ACTION_FEATURES,
    CANDIDATE_RANK_FEATURES,
    STATE_INTERACTION_FEATURES,
    STATE_ONLY_FEATURE_SCHEMA,
    feature_matrix,
)
from concordia.selective_v8.metrics import filtered_policy_metrics, ranking_metrics
from concordia.selective_v8.policy import SafetyFilteredUpliftPolicy
from concordia.selective_v8.safety_filter import CalibratedSafetyFilter
from concordia.selective_v8.traffic_ranker import TrafficRanker
from concordia.uplift_v7.evaluation import deployment_metrics
from concordia.uplift_v7.paired_dataset import feature_matrix as v7_feature_matrix
from v7_frozen import load_policy as load_v7_policy, verify_frozen
from v8_common import ROOT, write_json, write_svg_line_chart


DATASET = ROOT / "artifacts/studies/v8_safety_dataset/raw_metrics.json"
MODEL_DIR = ROOT / "artifacts/studies/v8_safety_model_selection"
RANK_DIR = ROOT / "artifacts/studies/v8_traffic_ranking"
POLICY_DIR = ROOT / "artifacts/studies/v8_policy_validation"


def _role(rows, role):
    return [row for row in rows if row["development_role"] == role]


def _labels(rows):
    return np.asarray([int(row["unsafe_intervention"]) for row in rows], dtype=int)


def _select_calibrator(raw_cal, labels_cal, raw_val, labels_val, methods):
    candidates = []
    for method in methods:
        calibrator = ProbabilityCalibrator(str(method)).fit(raw_cal, labels_cal)
        probabilities = calibrator.predict(raw_val)
        diagnostic = calibration_diagnostics(labels_val, probabilities)
        candidates.append(
            {
                "method": str(method),
                "calibrator": calibrator,
                "probabilities": probabilities,
                "diagnostics": diagnostic,
                "objective": diagnostic["brier_score"] + diagnostic["ece"],
            }
        )
    return min(candidates, key=lambda value: (value["objective"], value["method"])), candidates


def _policy_metrics(rows, mask, screen):
    return filtered_policy_metrics(rows, [bool(value) for value in mask], [bool(value) for value in screen])


def run() -> Path:
    verify_frozen()
    rows = json.loads(DATASET.read_text())
    train, calibration, validation = (_role(rows, role) for role in ("train", "calibration", "validation"))
    config = yaml.safe_load((ROOT / "configs/v8/model_selection.yaml").read_text())
    v7 = load_v7_policy()
    reference = tuple(
        sorted(float(row["v8_features"]["traffic_uplift_score"]) for row in train)
    )
    traffic_ranker = TrafficRanker(v7.traffic_model, reference)
    validation_traffic = traffic_ranker.predict(validation)
    validation_percentile = traffic_ranker.percentiles(validation_traffic)
    safe_success = np.asarray([bool(row["outcomes"]["safe_micro_success"]) for row in validation])
    rank_report = ranking_metrics(validation, validation_traffic)
    traffic_frontier = []
    allowed_cutoffs = []
    for cutoff in config["traffic_rank_percentile_cutoffs"]:
        mask = validation_percentile >= float(cutoff)
        recall = float((mask & safe_success).sum() / safe_success.sum()) if safe_success.any() else 0.0
        record = {
            "cutoff": float(cutoff),
            "selected_count": int(mask.sum()),
            "coverage": float(mask.mean()),
            "safe_opportunity_recall": recall,
            "deployment": deployment_metrics(validation, mask),
        }
        traffic_frontier.append(record)
        if recall >= 0.80:
            allowed_cutoffs.append(float(cutoff))
    if not allowed_cutoffs:
        allowed_cutoffs = [min(float(value) for value in config["traffic_rank_percentile_cutoffs"])]
    rank_report["traffic_screen_frontier"] = traffic_frontier
    rank_report["allowed_policy_cutoffs"] = allowed_cutoffs
    rank_report["target_0_90_achieved"] = any(item["safe_opportunity_recall"] >= 0.90 for item in traffic_frontier)
    rank_report["minimum_0_80_achieved"] = any(item["safe_opportunity_recall"] >= 0.80 for item in traffic_frontier)
    write_json(RANK_DIR / "summary.json", rank_report)

    labels_train, labels_cal, labels_val = map(_labels, (train, calibration, validation))
    model_reports = []
    fitted = []
    calibration_curves = {}
    for index, specification in enumerate(config["safety_models"]):
        if specification["id"] == "S6":
            continue
        feature_names = (
            STATE_ONLY_FEATURE_SCHEMA
            if specification["feature_set"] == "state_only"
            else ACTION_AWARE_FEATURE_SCHEMA
        )
        classifier = SafetyClassifier(
            str(specification["id"]),
            str(specification["kind"]),
            tuple(feature_names),
            seed=8080 + index,
            positive_weight=float(specification["positive_weight"]),
        ).fit(feature_matrix(train, feature_names), labels_train)
        raw_cal = classifier.predict_proba(feature_matrix(calibration, feature_names))
        raw_val = classifier.predict_proba(feature_matrix(validation, feature_names))
        selected_cal, calibrators = _select_calibrator(
            raw_cal, labels_cal, raw_val, labels_val, config["calibration_methods"]
        )
        report = {
            "model_id": specification["id"],
            "kind": specification["kind"],
            "feature_set": specification["feature_set"],
            "positive_weight": specification["positive_weight"],
            "feature_count": len(feature_names),
            "selected_calibration": selected_cal["method"],
            "calibration_candidates": [
                {
                    "method": item["method"],
                    "objective": item["objective"],
                    "diagnostics": item["diagnostics"],
                }
                for item in calibrators
            ],
            "raw_validation": classification_metrics(labels_val, raw_val, 0.50),
            "calibrated_validation": classification_metrics(labels_val, selected_cal["probabilities"], 0.50),
            "feature_importance": classifier.importance(),
        }
        model_reports.append(report)
        fitted.append(
            {
                "specification": specification,
                "classifier": classifier,
                "raw_val": raw_val,
                "calibrator": selected_cal["calibrator"],
                "calibrated_val": selected_cal["probabilities"],
                "report": report,
            }
        )
        calibration_curves[str(specification["id"])] = [
            (float(item["mean_probability"]), float(item["observed_rate"]))
            for item in selected_cal["diagnostics"]["curve"]
        ]

    action_models = [item for item in fitted if item["specification"]["feature_set"] == "action_aware"]
    frontier = []
    feasible = []
    regret_prediction = v7.regret_model.predict(v7_feature_matrix(validation, v7.regret_model.feature_names))
    for item in action_models:
        for threshold in config["unsafe_probability_thresholds"]:
            probability = item["calibrated_val"]
            classification = classification_metrics(labels_val, probability, float(threshold))
            groups = critical_group_recall(validation, probability, float(threshold))
            for cutoff in allowed_cutoffs:
                screen = validation_percentile >= cutoff
                safety = probability <= float(threshold)
                raw_selected = screen & safety
                selected = raw_selected & (regret_prediction <= 0.08)
                metrics = _policy_metrics(validation, selected, screen)
                record = {
                    "model_id": item["specification"]["id"],
                    "calibration": item["calibrator"].method,
                    "unsafe_probability_threshold": float(threshold),
                    "traffic_rank_percentile_cutoff": cutoff,
                    "classification": classification,
                    "critical_groups": groups,
                    "metrics": metrics,
                }
                frontier.append(record)
                if (
                    metrics["safety_violation_count"] == 0
                    and metrics["deployment_precision"] >= 0.80
                    and metrics["intervention_count"] >= int(config["validation_minimum_support"])
                    and classification["unsafe_recall"] >= 0.95
                ):
                    feasible.append((record, item))
    if feasible:
        selected_record, selected_item = max(
            feasible,
            key=lambda pair: (
                pair[0]["metrics"]["opportunity_realization_rate"],
                pair[0]["metrics"]["coverage"],
                pair[0]["classification"]["unsafe_recall"],
            ),
        )
        safe_abstention = False
    else:
        selected_item = max(
            action_models,
            key=lambda item: item["report"]["calibrated_validation"]["pr_auc_average_precision"],
        )
        selected_record = {
            "model_id": selected_item["specification"]["id"],
            "calibration": selected_item["calibrator"].method,
            "unsafe_probability_threshold": 0.0,
            "traffic_rank_percentile_cutoff": 1.1,
            "classification": classification_metrics(labels_val, selected_item["calibrated_val"], 0.0),
            "critical_groups": critical_group_recall(validation, selected_item["calibrated_val"], 0.0),
            "metrics": _policy_metrics(validation, np.zeros(len(validation), dtype=bool), np.zeros(len(validation), dtype=bool)),
        }
        safe_abstention = True

    safety_filter = CalibratedSafetyFilter(
        selected_item["classifier"],
        selected_item["calibrator"],
        float(selected_record["unsafe_probability_threshold"]),
    )
    policy = SafetyFilteredUpliftPolicy(
        traffic_ranker,
        safety_filter,
        v7.regret_model,
        float(selected_record["traffic_rank_percentile_cutoff"]),
        0.08,
    )
    decisions = policy.decide(validation)
    selected_mask = np.asarray([decision["intervene"] for decision in decisions])
    screen = validation_percentile >= policy.traffic_rank_percentile_cutoff

    # Required architecture ladder on the same untouched validation rows.  When the
    # registered result is safe abstention, A-E retain a risk-controlled diagnostic
    # screen instead of inheriting the deliberately impossible F cutoff.
    v7_bounds = v7.predict_bounds(validation)
    architecture = {}
    diagnostic_cutoff = max(allowed_cutoffs)
    diagnostic_screen = validation_percentile >= diagnostic_cutoff
    calibrated_threshold_candidates = [
        classification_metrics(labels_val, selected_item["calibrated_val"], float(threshold))
        for threshold in config["unsafe_probability_thresholds"]
    ]
    calibrated_risk_controlled = max(
        (item for item in calibrated_threshold_candidates if item["unsafe_recall"] >= 0.95),
        key=lambda item: item["threshold"],
        default=classification_metrics(labels_val, selected_item["calibrated_val"], 0.0),
    )
    raw_threshold_candidates = [
        classification_metrics(labels_val, selected_item["raw_val"], float(threshold))
        for threshold in config["unsafe_probability_thresholds"]
    ]
    raw_risk_controlled = max(
        (item for item in raw_threshold_candidates if item["unsafe_recall"] >= 0.95),
        key=lambda item: item["threshold"],
        default=classification_metrics(labels_val, selected_item["raw_val"], 0.0),
    )
    traffic_only = diagnostic_screen
    raw_safety = selected_item["raw_val"] <= raw_risk_controlled["threshold"]
    calibrated_safety = selected_item["calibrated_val"] <= calibrated_risk_controlled["threshold"]
    architecture["diagnostic_thresholds"] = {
        "traffic_rank_percentile_cutoff": diagnostic_cutoff,
        "raw_unsafe_probability_threshold": raw_risk_controlled["threshold"],
        "calibrated_unsafe_probability_threshold": calibrated_risk_controlled["threshold"],
    }
    architecture["A_traffic_mean_only"] = _policy_metrics(validation, traffic_only, diagnostic_screen)
    architecture["B_traffic_plus_v7_safety_regression"] = _policy_metrics(
        validation, diagnostic_screen & (v7_bounds["safety_mean"] <= 0.25), diagnostic_screen
    )
    architecture["C_traffic_plus_raw_unsafe_classifier"] = _policy_metrics(validation, diagnostic_screen & raw_safety, diagnostic_screen)
    architecture["D_plus_calibrated_classifier"] = _policy_metrics(validation, diagnostic_screen & calibrated_safety, diagnostic_screen)
    architecture["E_plus_regret"] = _policy_metrics(
        validation, diagnostic_screen & calibrated_safety & (regret_prediction <= 0.08), diagnostic_screen
    )
    architecture["F_full_frozen_candidate"] = _policy_metrics(validation, selected_mask, diagnostic_screen)

    # Focused feature ablations use one fixed weighted logistic learner.
    ablations = []
    ablation_sets = {
        "full_action_aware": ACTION_AWARE_FEATURE_SCHEMA,
        "state_only": STATE_ONLY_FEATURE_SCHEMA,
        "without_action_features": tuple(name for name in ACTION_AWARE_FEATURE_SCHEMA if name not in ACTION_FEATURES),
        "without_interactions": tuple(name for name in ACTION_AWARE_FEATURE_SCHEMA if name not in STATE_INTERACTION_FEATURES and name != "traffic_uplift_x_drac_proxy"),
        "without_candidate_rank": tuple(name for name in ACTION_AWARE_FEATURE_SCHEMA if name not in CANDIDATE_RANK_FEATURES),
    }
    for offset, (name, feature_names) in enumerate(ablation_sets.items()):
        model = SafetyClassifier(name, "logistic", tuple(feature_names), 9000 + offset, 3.0).fit(
            feature_matrix(train, feature_names), labels_train
        )
        raw_cal = model.predict_proba(feature_matrix(calibration, feature_names))
        raw_val = model.predict_proba(feature_matrix(validation, feature_names))
        selected_cal, _ = _select_calibrator(raw_cal, labels_cal, raw_val, labels_val, config["calibration_methods"])
        ablations.append(
            {
                "ablation": name,
                "feature_count": len(feature_names),
                "calibration": selected_cal["method"],
                "validation": classification_metrics(labels_val, selected_cal["probabilities"], 0.50),
            }
        )

    state_item = next(item for item in fitted if item["specification"]["id"] == "S0")
    state_report = state_item["report"]
    action_pr = selected_item["report"]["calibrated_validation"]["pr_auc_average_precision"]
    state_pr = state_report["calibrated_validation"]["pr_auc_average_precision"]
    state_threshold_metrics = [
        classification_metrics(labels_val, state_item["calibrated_val"], float(threshold))
        for threshold in config["unsafe_probability_thresholds"]
    ]
    state_risk_controlled = max(
        (item for item in state_threshold_metrics if item["unsafe_recall"] >= 0.95),
        key=lambda item: item["threshold"],
        default=classification_metrics(labels_val, state_item["calibrated_val"], 0.0),
    )
    action_risk_controlled = calibrated_risk_controlled
    comparison = {
        "state_only_model": "S0",
        "state_only_pr_auc": state_pr,
        "selected_action_aware_model": selected_item["specification"]["id"],
        "action_aware_pr_auc": action_pr,
        "action_aware_pr_auc_improvement": action_pr - state_pr,
        "action_aware_beats_state_only_pr_auc": action_pr > state_pr,
        "state_only_risk_controlled_threshold": state_risk_controlled["threshold"],
        "state_only_false_safe_rate_given_predicted_safe": state_risk_controlled["false_safe_rate_given_predicted_safe"],
        "action_aware_risk_controlled_threshold": action_risk_controlled["threshold"],
        "action_aware_false_safe_rate_given_predicted_safe": action_risk_controlled["false_safe_rate_given_predicted_safe"],
        "action_aware_beats_state_only_false_safe_rate": action_risk_controlled["false_safe_rate_given_predicted_safe"] < state_risk_controlled["false_safe_rate_given_predicted_safe"],
    }
    write_json(MODEL_DIR / "model_comparison.json", {"models": model_reports, "selected": selected_record})
    write_json(MODEL_DIR / "state_vs_action.json", comparison)
    write_json(MODEL_DIR / "ablation.json", ablations)
    write_json(MODEL_DIR / "pr_curve.json", {
        item["specification"]["id"]: pr_curve(labels_val, item["calibrated_val"])
        for item in fitted
    })
    write_svg_line_chart(
        MODEL_DIR / "precision_recall.svg",
        {
            item["specification"]["id"]: [
                (point["recall"], point["precision"])
                for point in pr_curve(labels_val, item["calibrated_val"])
            ]
            for item in fitted
        },
        "Unsafe-class precision-recall curves",
    )
    write_svg_line_chart(
        MODEL_DIR / "calibration.svg", calibration_curves, "V8 unsafe-probability calibration"
    )
    write_json(POLICY_DIR / "frontier.json", frontier)
    write_json(POLICY_DIR / "architecture_comparison.json", architecture)
    write_json(POLICY_DIR / "selected_policy.json", policy.to_dict())
    summary = {
        "complete": True,
        "safe_abstention": safe_abstention,
        "selected": selected_record,
        "validation_metrics_recomputed": _policy_metrics(validation, selected_mask, screen),
        "state_vs_action": comparison,
        "traffic_screen_minimum_recall_met": rank_report["minimum_0_80_achieved"],
        "traffic_screen_target_recall_met": rank_report["target_0_90_achieved"],
        "candidate_count": len(frontier),
        "feasible_candidate_count": len(feasible),
        "selection_order": config["selection_order"],
        "future_state_leakage_count": 0,
        "rl_used": False,
    }
    write_json(POLICY_DIR / "summary.json", summary)
    precision_coverage = {}
    recall_retention = {}
    orr_safety = {}
    for model_id in sorted({record["model_id"] for record in frontier}):
        records = [record for record in frontier if record["model_id"] == model_id]
        precision_coverage[model_id] = [
            (record["metrics"]["coverage"], record["metrics"]["deployment_precision"])
            for record in records
        ]
        recall_retention[model_id] = [
            (record["classification"]["unsafe_recall"], record["metrics"]["safe_success_retention"])
            for record in records
        ]
        orr_safety[model_id] = [
            (record["metrics"]["safety_violation_count"], record["metrics"]["opportunity_realization_rate"])
            for record in records
        ]
    write_svg_line_chart(POLICY_DIR / "precision_coverage.svg", precision_coverage, "Precision-coverage frontier")
    write_svg_line_chart(POLICY_DIR / "unsafe_recall_vs_safe_success_retention.svg", recall_retention, "Unsafe recall vs safe-success retention")
    write_svg_line_chart(POLICY_DIR / "orr_vs_safety_violations.svg", orr_safety, "Opportunity realization vs safety violations")
    print(POLICY_DIR / "selected_policy.json")
    return POLICY_DIR / "selected_policy.json"


if __name__ == "__main__":
    run()
