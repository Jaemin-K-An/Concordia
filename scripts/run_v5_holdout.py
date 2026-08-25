#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

from concordia.evaluation import summarize_selective_policy
from concordia.feasibility import (
    V4PredictionBundle,
    V4_FEATURE_SCHEMA,
    build_alignment_case,
    expand_v4_features,
    unified_calibration_metrics,
)
from concordia.selective import (
    PrecisionConstrainedPolicy,
    V4DecisionInputs,
    V5DecisionInputs,
)
from v5_frozen import analytical_policy, load_deployment, prepare_cases, verify_frozen, write_json


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/studies/v5_frozen_holdout"


def _v4_predictions(cases: list[dict]):
    study = ROOT / "artifacts/studies/v4_precision_validation"
    bundle = V4PredictionBundle.from_packages(
        json.loads((study / "probability_package.json").read_text()),
        json.loads((study / "benefit_package.json").read_text()),
        json.loads((study / "safety_package.json").read_text()),
    )
    matrix = np.asarray(
        [
            [expand_v4_features(case)[name] for name in V4_FEATURE_SCHEMA]
            for case in cases
        ],
        dtype=float,
    )
    prediction = bundle.predict(matrix)
    selection = json.loads((study / "threshold_selection.json").read_text())
    point = selection["policy_operating_points"]["V4-P"]
    policy = PrecisionConstrainedPolicy(
        "V4-P",
        probability_threshold=float(point["score_threshold"]),
        benefit_threshold=float(point["benefit_threshold"]),
        safety_delta=float(selection["safety_delta"]),
        safety_probability_threshold=float(
            selection["safety_failure_probability_threshold"]
        ),
    )
    return prediction, policy


def _policy_row(case: dict, intervene: bool, name: str) -> dict:
    baseline = float(case["baseline_metrics"]["eta_only_ttt"])
    adaptive = float(case["adaptive_counterfactual"]["ttt"])
    success = case["label"] == "WIN"
    return {
        "case_id": case["case_id"],
        "policy": name,
        "intervene": bool(intervene),
        "success": bool(intervene and success),
        "counterfactual_success": success,
        "system_ttt_gain": baseline - adaptive if intervene else 0.0,
        "relative_ttt_gain": (baseline - adaptive) / max(baseline, 1e-9)
        if intervene
        else 0.0,
        "regret_violation": bool(
            intervene and case["adaptive_counterfactual"]["maximum_regret"] > 0.08
        ),
        "safety_violation": bool(
            intervene and case["adaptive_counterfactual"]["safety_difference"] > 0.25
        ),
        "legal_violation": bool(
            intervene and not case["adaptive_counterfactual"]["legal"]
        ),
        "regime": case["regime"],
        "shift_class": case["shift_class"],
        "scenario": case["scenario"],
        "navigation_penetration": case["condition"]["navigation_penetration"],
    }


def _group_metrics(rows: list[dict]) -> dict:
    result = {}
    critical_precision = []
    for dimension in (
        "regime",
        "shift_class",
        "scenario",
        "navigation_penetration",
    ):
        groups = defaultdict(list)
        for row in rows:
            groups[str(row[dimension])].append(row)
        result[dimension] = {}
        for key, values in sorted(groups.items()):
            metrics = summarize_selective_policy(values)
            result[dimension][key] = metrics
            if metrics["intervention_count"] >= 10:
                critical_precision.append(metrics["intervention_precision"])
    result["worst_critical_group_precision"] = (
        min(critical_precision) if critical_precision else None
    )
    return result


def run() -> Path:
    existing = OUTPUT / "summary.json"
    if existing.is_file():
        verify_frozen()
        print(existing)
        return existing
    manifest = verify_frozen()
    split = yaml.safe_load((ROOT / "configs/v5/splits.yaml").read_text())
    prereg = yaml.safe_load((ROOT / "configs/v5/preregistration.yaml").read_text())
    spec = split["final_analytical_holdout"]
    cases = []
    for scenario in spec["scenarios"]:
        for seed in spec["seeds"]:
            for demand in spec["demand_scale"]:
                for heterogeneity in spec["heterogeneity"]:
                    for penetration in spec["navigation_penetration"]:
                        cases.append(
                            build_alignment_case(
                                scenario=str(scenario),
                                seed=int(seed),
                                demand_scale=float(demand),
                                heterogeneity=str(heterogeneity),
                                navigation_penetration=float(penetration),
                                user_count=6,
                                regret_limit=0.08,
                                epsilon_grid=[0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.24],
                                minimum_relative_ttt_gain=0.01,
                                safety_delta=0.25,
                                source_split="v5_final_analytical_holdout",
                            )
                        )
    if len(cases) != int(spec["expected_case_count"]):
        raise RuntimeError("v5 analytical holdout count mismatch")
    development = json.loads(
        (ROOT / "artifacts/studies/v5_model_selection/split_manifest.json").read_text()
    )["roles"]
    development_ids = {case_id for values in development.values() for case_id in values}
    if {case["case_id"] for case in cases} & development_ids:
        raise RuntimeError("v5 analytical holdout leaked into development")
    regime, shift, shift_names, bundle, thresholds = load_deployment()
    started = time.perf_counter()
    rows, prediction = prepare_cases(cases, regime, shift, shift_names, bundle)
    latency = (time.perf_counter() - started) / len(rows)
    variants = {
        name: analytical_policy(thresholds, variant=name)
        for name in ("V5-G", "V5-R", "V5-RD", "V5-RS", "V5-F")
    }
    policy_rows = {name: [] for name in ("B6", "V4-P", *variants)}
    decisions = []
    for index, row in enumerate(rows):
        inputs = V5DecisionInputs(
            row["case_id"],
            row["regime"],
            row["shift_class"],
            row["domain_shift_score"],
            float(prediction.success_probability[index]),
            float(prediction.analytical_benefit[index]),
            float(prediction.corrected_microscopic_benefit[index]),
            float(prediction.microscopic_success_probability[index]),
            float(prediction.microscopic_safety_probability_upper[index]),
            bool(row["adaptive_counterfactual"]["legal"]),
        )
        for name, policy in variants.items():
            decision = policy.decide(inputs)
            policy_rows[name].append(_policy_row(row, decision.intervene, name))
            if name == "V5-RD":
                decisions.append(
                    {
                        "case_id": row["case_id"],
                        "intervene": decision.intervene,
                        "reasons": list(decision.reasons),
                        "explanation": list(decision.explanation),
                        "computed_values": dict(decision.computed_values),
                        "regime": row["regime"],
                        "shift_class": row["shift_class"],
                    }
                )
        policy_rows["B6"].append(_policy_row(row, True, "B6"))
    v4_prediction, v4_policy = _v4_predictions(cases)
    for index, row in enumerate(rows):
        inputs = V4DecisionInputs(
            row["case_id"],
            float(v4_prediction["probability"][index]),
            float(v4_prediction["probability_lower"][index]),
            float(v4_prediction["expected_benefit"][index]),
            float(v4_prediction["benefit_lower"][index]),
            float(v4_prediction["safety_difference_upper"][index]),
            float(v4_prediction["safety_failure_probability"][index]),
            float(v4_prediction["safety_failure_probability_upper"][index]),
            float(v4_prediction["esiv"][index]),
            float(v4_prediction["esiv_lower"][index]),
            bool(row["adaptive_counterfactual"]["legal"]),
        )
        policy_rows["V4-P"].append(
            _policy_row(row, v4_policy.decide(inputs).intervene, "V4-P")
        )
    metrics = {
        name: summarize_selective_policy(values) for name, values in policy_rows.items()
    }
    primary = metrics["V5-RD"]
    groups = _group_metrics(policy_rows["V5-RD"])
    false_positives = [
        (case, decision)
        for case, decision in zip(rows, decisions)
        if decision["intervene"] and case["label"] != "WIN"
    ]
    failures = Counter()
    for case, _decision in false_positives:
        if case["adaptive_counterfactual"]["safety_difference"] > 0.25:
            failures["safety_mismatch"] += 1
        elif case["features"]["acceptance_probability"] < 0.5:
            failures["partial_adoption_feedback"] += 1
        elif case["shift_class"] != "IN_DISTRIBUTION":
            failures["regime_or_domain_shift"] += 1
        elif case["regime"] == "STRUCTURALLY_CONSTRAINED":
            failures["structural_topology_failure"] += 1
        else:
            failures["benefit_prediction_error"] += 1
    if primary["intervention_precision"] < 0.80:
        provisional = "F"
    elif (
        primary["coverage"] >= 0.20
        and primary["intervention_count"] >= prereg["primary"]["analytical_interventions_minimum"]
    ):
        provisional = "S+_PENDING_MICRO"
    elif (
        primary["coverage"] >= 0.15
        and primary["intervention_count"] >= prereg["primary"]["analytical_interventions_minimum"]
    ):
        provisional = "S_PENDING_MICRO"
    else:
        provisional = "P"
    summary = {
        "complete": True,
        "study": "v5 frozen analytical holdout",
        "untouched_holdout": True,
        "case_count": len(rows),
        "label_counts": dict(Counter(row["label"] for row in rows)),
        "regime_counts": dict(Counter(row["regime"] for row in rows)),
        "shift_counts": dict(Counter(row["shift_class"] for row in rows)),
        "selected_policy": "V5-RD",
        "policy_metrics": metrics,
        "primary_metrics": primary,
        "group_metrics": groups,
        "calibration_metrics": unified_calibration_metrics(
            [int(row["label"] == "WIN") for row in rows],
            prediction.success_probability,
        ),
        "failure_taxonomy": dict(failures),
        "provisional_outcome": provisional,
        "mean_decision_latency_seconds": latency,
        "holdout_absent_from_development": True,
        "freeze_source_commit": manifest["source_commit"],
        "frozen_immutable": True,
        "rl_used": False,
        "claim_boundary": prereg["claim_boundary"],
    }
    write_json(OUTPUT / "raw_metrics.json", rows)
    write_json(OUTPUT / "policy_rows.json", policy_rows)
    write_json(OUTPUT / "decision_log.json", decisions)
    write_json(OUTPUT / "summary.json", summary)
    print(OUTPUT / "summary.json")
    return OUTPUT / "summary.json"


if __name__ == "__main__":
    run()
