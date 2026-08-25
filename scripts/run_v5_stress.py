#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
from pathlib import Path

import yaml

from concordia.evaluation import summarize_selective_policy
from concordia.feasibility import build_alignment_case
from concordia.selective import V5DecisionInputs
from build_v5_dataset import _adjust_case
from v5_frozen import analytical_policy, load_deployment, prepare_cases, verify_frozen, write_json


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/studies/v5_stress_holdout"


def _row(case: dict, intervene: bool, policy: str) -> dict:
    baseline = float(case["baseline_metrics"]["eta_only_ttt"])
    adaptive = float(case["adaptive_counterfactual"]["ttt"])
    success = case["label"] == "WIN"
    return {
        "case_id": case["case_id"],
        "policy": policy,
        "intervene": intervene,
        "success": bool(intervene and success),
        "counterfactual_success": success,
        "system_ttt_gain": baseline - adaptive if intervene else 0.0,
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
    }


def run() -> Path:
    existing = OUTPUT / "summary.json"
    if existing.is_file():
        verify_frozen()
        print(existing)
        return existing
    verify_frozen()
    split = yaml.safe_load((ROOT / "configs/v5/splits.yaml").read_text())
    stress = yaml.safe_load((ROOT / "configs/v5/stress.yaml").read_text())
    spec = split["final_stress_holdout"]
    cases = []
    index = 0
    for scenario in spec["scenarios"]:
        for seed in spec["seeds"]:
            for demand in spec["demand_scale"]:
                for heterogeneity in spec["heterogeneity"]:
                    for penetration in spec["navigation_penetration"]:
                        acceptance = float(
                            spec["acceptance_multiplier"][
                                index % len(spec["acceptance_multiplier"])
                            ]
                        )
                        variance = float(
                            spec["preference_variance_multiplier"][
                                (index // len(spec["acceptance_multiplier"]))
                                % len(spec["preference_variance_multiplier"])
                            ]
                        )
                        index += 1
                        case = build_alignment_case(
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
                            source_split="v5_final_stress_holdout",
                        )
                        case = _adjust_case(case, acceptance, variance)
                        case["case_id"] += f"-a{acceptance:.2f}-v{variance:.2f}"
                        cases.append(case)
    if len(cases) != int(spec["expected_case_count"]):
        raise RuntimeError("v5 stress holdout count mismatch")
    regime, shift, shift_names, bundle, thresholds = load_deployment()
    rows, prediction = prepare_cases(cases, regime, shift, shift_names, bundle)
    policies = {
        "V5-R_no_DSS": analytical_policy(thresholds, variant="V5-R"),
        "V5-RD": analytical_policy(thresholds, variant="V5-RD"),
    }
    policy_rows = {name: [] for name in policies}
    decision_log = []
    for index, case in enumerate(rows):
        inputs = V5DecisionInputs(
            case["case_id"],
            case["regime"],
            case["shift_class"],
            case["domain_shift_score"],
            float(prediction.success_probability[index]),
            float(prediction.analytical_benefit[index]),
            float(prediction.corrected_microscopic_benefit[index]),
            float(prediction.microscopic_success_probability[index]),
            float(prediction.microscopic_safety_probability_upper[index]),
            bool(case["adaptive_counterfactual"]["legal"]),
        )
        for name, policy in policies.items():
            decision = policy.decide(inputs)
            policy_rows[name].append(_row(case, decision.intervene, name))
            if name == "V5-RD":
                decision_log.append(
                    {
                        "case_id": case["case_id"],
                        "intervene": decision.intervene,
                        "reasons": list(decision.reasons),
                        "explanation": list(decision.explanation),
                        "regime": case["regime"],
                        "shift_class": case["shift_class"],
                        "domain_shift_score": case["domain_shift_score"],
                    }
                )
    metrics = {
        name: summarize_selective_policy(values) for name, values in policy_rows.items()
    }
    primary = metrics["V5-RD"]
    target_met = (
        primary["intervention_precision"] >= float(stress["precision_target"])
        and primary["safety_violation_count"] == int(stress["safety_violations_target"])
        and primary["coverage"] > 0.0
    )
    summary = {
        "complete": True,
        "study": "v5 frozen unseen secondary stress holdout",
        "untouched_holdout": True,
        "case_count": len(rows),
        "label_counts": dict(Counter(row["label"] for row in rows)),
        "regime_counts": dict(Counter(row["regime"] for row in rows)),
        "shift_counts": dict(Counter(row["shift_class"] for row in rows)),
        "policy_metrics": metrics,
        "primary_metrics": primary,
        "stress_target_met": target_met,
        "H22_DSS_improves_shift_safety": {
            "supported": primary["intervention_precision"]
            >= metrics["V5-R_no_DSS"]["intervention_precision"]
            and primary["safety_violation_count"]
            <= metrics["V5-R_no_DSS"]["safety_violation_count"],
            "coverage_change": primary["coverage"]
            - metrics["V5-R_no_DSS"]["coverage"],
        },
        "rl_used": False,
        "claim_boundary": stress["claim_boundary"],
    }
    write_json(OUTPUT / "raw_metrics.json", rows)
    write_json(OUTPUT / "policy_rows.json", policy_rows)
    write_json(OUTPUT / "decision_log.json", decision_log)
    write_json(OUTPUT / "summary.json", summary)
    print(OUTPUT / "summary.json")
    return OUTPUT / "summary.json"


if __name__ == "__main__":
    run()
