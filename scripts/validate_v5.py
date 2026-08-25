#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from concordia.evaluation import summarize_selective_policy
from concordia.feasibility import V5ModelBundle, V5_FEATURE_SCHEMA
from concordia.selective import RegimeConditionedPolicy, V5DecisionInputs


ROOT = Path(__file__).resolve().parents[1]
MODEL_STUDY = ROOT / "artifacts/studies/v5_model_selection"
SHIFT_STUDY = ROOT / "artifacts/studies/v5_shift_detection"
MICRO_STUDY = ROOT / "artifacts/studies/v5_micro_calibration"
OUTPUT = ROOT / "artifacts/studies/v5_policy_validation"


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _matrix(rows: list[dict]) -> np.ndarray:
    return np.asarray(
        [[row["features"][name] for name in V5_FEATURE_SCHEMA] for row in rows],
        dtype=float,
    )


def _select_thresholds(
    rows: list[dict], probability: np.ndarray, candidates: list[float]
) -> tuple[dict, dict]:
    labels = np.asarray([row["targets"]["success"] for row in rows], dtype=int)
    regimes = sorted({row["regime"] for row in rows})
    table: dict[str, dict[str, float | None]] = {}
    evidence = {}
    for regime in regimes:
        table[regime] = {}
        regime_rows = np.asarray([row["regime"] == regime for row in rows])
        for shift_class in ("IN_DISTRIBUTION", "MILD_SHIFT", "STRONG_SHIFT"):
            if shift_class == "STRONG_SHIFT":
                table[regime][shift_class] = None
                evidence[f"{regime}::{shift_class}"] = {
                    "selected_threshold": None,
                    "reason": "preregistered strong-shift abstention",
                }
                continue
            cell = regime_rows & np.asarray(
                [row["shift_class"] == shift_class for row in rows]
            )
            if int(cell.sum()) < 10 and shift_class == "MILD_SHIFT":
                inherited = table[regime].get("IN_DISTRIBUTION")
                table[regime][shift_class] = (
                    min(0.99, float(inherited) + 0.05)
                    if inherited is not None
                    else None
                )
                evidence[f"{regime}::{shift_class}"] = {
                    "case_count": int(cell.sum()),
                    "selected_threshold": table[regime][shift_class],
                    "reason": "sparse cell inherits conservative in-distribution threshold",
                }
                continue
            curve = []
            for threshold in candidates:
                selected = cell & (probability >= threshold)
                count = int(selected.sum())
                curve.append(
                    {
                        "threshold": float(threshold),
                        "intervention_count": count,
                        "precision": float(labels[selected].mean()) if count else 0.0,
                        "coverage_within_cell": count / max(1, int(cell.sum())),
                    }
                )
            feasible = [value for value in curve if value["precision"] >= 0.80]
            chosen = max(
                feasible,
                key=lambda value: (value["intervention_count"], value["precision"]),
                default=None,
            )
            table[regime][shift_class] = (
                float(chosen["threshold"]) if chosen is not None else None
            )
            evidence[f"{regime}::{shift_class}"] = {
                "case_count": int(cell.sum()),
                "selected_threshold": table[regime][shift_class],
                "curve": curve,
                "precision_target_met": bool(chosen),
            }
    return table, evidence


def _policy_rows(
    name: str,
    source: list[dict],
    predictions,
    policy: RegimeConditionedPolicy,
) -> tuple[list[dict], dict]:
    output = []
    for index, row in enumerate(source):
        inputs = V5DecisionInputs(
            case_id=row["case_id"],
            regime=row["regime"],
            shift_class=row["shift_class"],
            domain_shift_score=float(row["domain_shift_score"]),
            success_probability=float(predictions.success_probability[index]),
            analytical_benefit=float(predictions.analytical_benefit[index]),
            corrected_microscopic_benefit=float(
                predictions.corrected_microscopic_benefit[index]
            ),
            microscopic_success_probability=float(
                predictions.microscopic_success_probability[index]
            ),
            microscopic_safety_probability_upper=float(
                predictions.microscopic_safety_probability_upper[index]
            ),
            legal=True,
        )
        decision = policy.decide(inputs)
        success = bool(row["targets"]["success"])
        output.append(
            {
                "case_id": row["case_id"],
                "policy": name,
                "intervene": decision.intervene,
                "success": success and decision.intervene,
                "counterfactual_success": success,
                "system_ttt_gain": float(row["targets"]["relative_ttt_gain"])
                if decision.intervene
                else 0.0,
                "regret_violation": False,
                "safety_violation": bool(
                    decision.intervene
                    and row["targets"]["safety_difference"] > 0.25
                ),
                "legal_violation": False,
                "regime": row["regime"],
                "shift_class": row["shift_class"],
                "scenario": row["scenario"],
                "navigation_penetration": row["condition"][
                    "navigation_penetration"
                ],
                "decision": {
                    "selected_policy": decision.selected_policy,
                    "reasons": list(decision.reasons),
                    "explanation": list(decision.explanation),
                    "computed_values": dict(decision.computed_values),
                },
            }
        )
    return output, summarize_selective_policy(output)


def _groups(rows: list[dict]) -> dict:
    result = {}
    for dimension in (
        "regime",
        "shift_class",
        "scenario",
        "navigation_penetration",
    ):
        groups = defaultdict(list)
        for row in rows:
            groups[str(row[dimension])].append(row)
        result[dimension] = {
            key: summarize_selective_policy(values)
            for key, values in sorted(groups.items())
        }
    return result


def _micro_threshold() -> tuple[float, list[dict]]:
    rows = json.loads((MICRO_STUDY / "raw_metrics.json").read_text())
    rows = [row for row in rows if row["development_role"] == "micro_calibration"]
    package = json.loads((MICRO_STUDY / "micro_correction_package.json").read_text())
    safety = json.loads((MICRO_STUDY / "micro_safety_package.json").read_text())
    from concordia.feasibility import (
        MICRO_ADDITIONAL_FEATURES,
        MicroscopicCorrectionModel,
        ProbabilityCalibrator,
    )
    from concordia.safety import MicroscopicSafetyVeto

    schema = V5_FEATURE_SCHEMA + MICRO_ADDITIONAL_FEATURES
    matrix = np.asarray(
        [[row["micro_features"][name] for name in schema] for row in rows]
    )
    model = MicroscopicCorrectionModel.from_dict(package["model"])
    calibrator = ProbabilityCalibrator.from_dict(package["calibrator"])
    veto = MicroscopicSafetyVeto.from_dict(safety["veto"])
    corrected, raw = model.predict(
        matrix, [row["analytical_benefit"] for row in rows]
    )
    probability = calibrator.predict(raw)
    _mean, upper = veto.predict(matrix)
    values = []
    for threshold in (0.40, 0.50, 0.60, 0.70, 0.80):
        selected = (
            (probability >= threshold)
            & (corrected > 0.0)
            & (upper < veto.probability_threshold)
        )
        count = int(selected.sum())
        success = sum(
            int(row["microscopic_success"])
            for row, flag in zip(rows, selected)
            if flag
        )
        unsafe = sum(
            int(row["microscopic_safety_violation"])
            for row, flag in zip(rows, selected)
            if flag
        )
        values.append(
            {
                "threshold": threshold,
                "intervention_count": count,
                "precision": success / max(1, count),
                "safety_violation_count": unsafe,
            }
        )
    feasible = [
        value
        for value in values
        if value["precision"] >= 0.50 and value["safety_violation_count"] == 0
    ]
    selected = max(
        feasible or values,
        key=lambda value: (
            value["precision"] >= 0.50,
            value["safety_violation_count"] == 0,
            value["intervention_count"],
            value["precision"],
        ),
    )
    return float(selected["threshold"]), values


def run() -> Path:
    existing = OUTPUT / "validation_summary.json"
    if (ROOT / "configs/v5/frozen_thresholds.yaml").is_file():
        if not existing.is_file():
            raise RuntimeError("v5 is frozen but validation evidence is missing")
        print(existing)
        return existing
    model_config = yaml.safe_load((ROOT / "configs/v5/model.yaml").read_text())
    rows = json.loads((MODEL_STUDY / "enriched_rows.json").read_text())
    rows = [row for row in rows if row["development_role"] == "validation"]
    analytical = json.loads((SHIFT_STUDY / "calibrated_model.json").read_text())
    micro = json.loads((MICRO_STUDY / "micro_correction_package.json").read_text())
    safety = json.loads((MICRO_STUDY / "micro_safety_package.json").read_text())
    bundle = V5ModelBundle.from_packages(analytical, micro, safety)
    predictions = bundle.predict(_matrix(rows), [row["regime"] for row in rows])
    thresholds, threshold_evidence = _select_thresholds(
        rows,
        predictions.success_probability,
        [float(value) for value in model_config["probability_thresholds"]],
    )
    micro_threshold, micro_curve = _micro_threshold()
    global_threshold = float(
        json.loads((SHIFT_STUDY / "calibration_summary.json").read_text())[
            "selected_threshold"
        ]
    )
    regimes = sorted(thresholds)
    global_table = {
        regime: {shift: global_threshold for shift in ("IN_DISTRIBUTION", "MILD_SHIFT", "STRONG_SHIFT")}
        for regime in regimes
    }
    no_shift_table = {
        regime: {
            shift: thresholds[regime]["IN_DISTRIBUTION"]
            for shift in ("IN_DISTRIBUTION", "MILD_SHIFT", "STRONG_SHIFT")
        }
        for regime in regimes
    }
    common = {
        "shift_probability_penalty": float(model_config["shift_probability_penalty"]),
        "micro_success_threshold": micro_threshold,
        "micro_safety_threshold": float(safety["veto"]["probability_threshold"]),
    }
    policies = {
        "V5-G": RegimeConditionedPolicy(
            global_table, **common, use_shift_gate=False, use_micro_correction=False, use_micro_safety_veto=False
        ),
        "V5-R": RegimeConditionedPolicy(
            no_shift_table, **common, use_shift_gate=False, use_micro_correction=False, use_micro_safety_veto=False
        ),
        "V5-RD": RegimeConditionedPolicy(
            thresholds, **common, use_shift_gate=True, use_micro_correction=False, use_micro_safety_veto=False
        ),
        "V5-RS": RegimeConditionedPolicy(
            thresholds, **common, use_shift_gate=True, use_micro_correction=False, use_micro_safety_veto=True
        ),
        "V5-F": RegimeConditionedPolicy(thresholds, **common),
    }
    all_rows = []
    metrics = {}
    groups = {}
    for name, policy in policies.items():
        policy_rows, policy_metrics = _policy_rows(name, rows, predictions, policy)
        all_rows.extend(policy_rows)
        metrics[name] = policy_metrics
        groups[name] = _groups(policy_rows)
    calibration_results = json.loads(
        (SHIFT_STUDY / "calibration_results.json").read_text()
    )
    best_by_model = {}
    for value in calibration_results:
        current = best_by_model.get(value["model"])
        if current is None or value["frontier"]["selected"]["coverage"] > current[
            "frontier"
        ]["selected"]["coverage"]:
            best_by_model[value["model"]] = value
    m1 = best_by_model["M1_global_logistic"]
    m2 = best_by_model["M2_interaction_logistic"]
    hypotheses = {
        "H21": {
            "supported_on_validation": metrics["V5-R"]["intervention_precision"] >= 0.80
            and metrics["V5-R"]["coverage"] >= metrics["V5-G"]["coverage"],
            "global": metrics["V5-G"],
            "regime_conditioned": metrics["V5-R"],
        },
        "H22": {"status": "reserved for frozen stress holdout"},
        "H23": {
            "status": "development evidence only",
            "without_veto": metrics["V5-RD"],
            "with_veto": metrics["V5-RS"],
        },
        "H24": {"status": "reserved for microscopic holdout"},
        "H25": {"status": "reserved for microscopic holdout"},
        "H26": {
            "supported_on_validation": metrics["V5-RD"]["coverage"]
            < metrics["V5-G"]["coverage"],
            "interpretation": "selectivity is an explicit safety and transfer mechanism",
        },
        "H27": {
            "selected_model": analytical["selected_model"],
            "selected_calibration": analytical["selected_calibration"],
            "candidate_count": len(calibration_results),
        },
        "H28": {
            "interaction_model_coverage_delta": m2["frontier"]["selected"]["coverage"]
            - m1["frontier"]["selected"]["coverage"],
            "interaction_model_brier_delta": m2["overall"]["brier_score"]
            - m1["overall"]["brier_score"],
            "supported_on_validation": m2["frontier"]["selected"]["coverage"]
            > m1["frontier"]["selected"]["coverage"]
            and m2["overall"]["brier_score"] < m1["overall"]["brier_score"],
        },
    }
    threshold_package = {
        "complete": True,
        "analytical_primary_policy": "V5-RD",
        "full_bridge_policy": "V5-F",
        "global_probability_threshold": global_threshold,
        "probability_thresholds": thresholds,
        "microscopic_probability_thresholds": {
            regime: {
                "IN_DISTRIBUTION": 0.0,
                "MILD_SHIFT": 0.0,
                "STRONG_SHIFT": None,
            }
            for regime in regimes
        },
        "shift_probability_penalty": common["shift_probability_penalty"],
        "microscopic_shift_probability_penalty": 0.0,
        "micro_success_threshold": micro_threshold,
        "micro_benefit_threshold": 0.0,
        "micro_safety_threshold": common["micro_safety_threshold"],
        "strong_shift_action": "abstain",
        "selection_roles": ["validation", "micro_calibration"],
        "final_holdouts_used": False,
    }
    _write(OUTPUT / "threshold_package.json", threshold_package)
    _write(OUTPUT / "threshold_selection.json", threshold_evidence)
    _write(OUTPUT / "micro_threshold_selection.json", micro_curve)
    _write(OUTPUT / "policy_rows.json", all_rows)
    _write(OUTPUT / "group_metrics.json", groups)
    _write(OUTPUT / "hypothesis_development_results.json", hypotheses)
    _write(
        existing,
        {
            "complete": True,
            "validation_case_count": len(rows),
            "policy_metrics": metrics,
            "analytical_primary": metrics["V5-RD"],
            "full_bridge_diagnostic": metrics["V5-F"],
            "final_holdouts_used": False,
        },
    )
    print(existing)
    return existing


if __name__ == "__main__":
    run()
