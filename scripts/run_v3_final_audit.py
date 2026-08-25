#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "FINAL_AUDIT_V3.md"
MACHINE = ROOT / "artifacts/final_audit_v3.json"


def _load(path: str):
    source = ROOT / path
    if not source.is_file():
        raise RuntimeError(f"required audit evidence is missing: {path}")
    return json.loads(source.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _status(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def run() -> Path:
    feasibility = _load("artifacts/studies/v3_feasibility_prediction/summary.json")
    holdout = _load("artifacts/studies/v3_selective_holdout/summary.json")
    micro = _load("artifacts/studies/v3_microscopic_selective/summary.json")
    real = _load("artifacts/studies/v3_real_topology_selective/summary.json")
    tail = _load("artifacts/studies/v3_tail_robustness/summary.json")
    freeze_manifest = _load("artifacts/v3/freeze_manifest.json")
    frozen_path = ROOT / "configs/v3/frozen_thresholds.yaml"
    frozen = yaml.safe_load(frozen_path.read_text(encoding="utf-8"))
    selected_path = ROOT / "artifacts/studies/v3_feasibility_prediction/selected_model.json"
    primary = holdout["primary_metrics"]
    calibration_ece = feasibility["validation_metrics"]["ece"]
    leakage_free = holdout["untouched_holdout"] and holdout[
        "holdout_case_ids_absent_from_training"
    ]
    frozen_before = (
        freeze_manifest["holdout_started"]
        and freeze_manifest["holdout_completed"]
        and freeze_manifest["threshold_config_hash"] == _sha(frozen_path)
        and frozen["selected_model_hash"] == _sha(selected_path)
        and holdout["threshold_immutable"]
    )
    safety_pass = primary["safety_violation_count"] == 0
    questions = {
        "Feasibility predictor calibrated?": (
            f"PARTIAL — calibration evaluated; ECE={calibration_ece:.3f}"
        ),
        "Leakage-free holdout?": _status(leakage_free),
        "Threshold frozen before holdout?": _status(frozen_before),
        "Intervention precision > 50%?": _status(primary["intervention_precision"] > 0.50),
        "Precision engineering target ≥65%?": _status(primary["intervention_precision"] >= 0.65),
        "Coverage ≥40%?": _status(primary["coverage"] >= 0.40),
        "Mean TTT gain positive?": _status(primary["mean_network_ttt_gain"] > 0),
        "Safety non-inferiority?": _status(safety_pass),
        "Real-topology selective policy tested?": _status(real["complete"] and real["evaluation_od_unseen"]),
        "Tail robustness evaluated?": _status(tail["complete"] and tail["threshold_immutable"]),
        "RL used?": "NO",
    }
    h = {
        "H8": _status(holdout["statistical_tests"]["H8_scientific_lower_ci_above_half"]),
        "H8_point": _status(holdout["statistical_tests"]["H8_point_precision_above_half"]),
        "H9": _status(holdout["statistical_tests"]["H9_failure_rate_reduced_vs_B6"]),
        "H10": _status(
            holdout["statistical_tests"]["H10_mean_network_cost_noninferior"]
            and tail["statistics"]["H10_tail_gate_pass"]
        ),
        "H11": _status(
            micro["statistics"]["H11_V3_safety_failure_count"]
            <= micro["statistics"]["H11_B6_safety_failure_count"]
        ),
        "H12": _status(
            real["statistics"]["H12_all_recommended_paths_legal"]
            and real["statistics"]["H12_unseen_od"]
            and real["statistics"]["H12_positive_net_benefit"]
        ),
        "H13": "DESCRIPTIVE",
        "H14": "DESCRIPTIVE",
    }
    outcome = holdout["outcome"]
    outcome_text = {
        "S": "Selective CONCORDIA supported.",
        "P": "Selective CONCORDIA partially supported.",
        "F": "Selective CONCORDIA not supported.",
    }[outcome]
    table = "\n".join(f"| {question} | **{status}** |" for question, status in questions.items())
    hypothesis_table = "\n".join(f"| {name} | **{status}** |" for name, status in h.items())
    text = f"""# CONCORDIA final audit v3

The v2 negative results remain unchanged: H1, H2, H3, H4, and H6 are not rewritten by this selective-policy study. Study V is development/validation evidence only; Study VI is the primary untouched holdout.

## Completion checks

| Question | Status |
|---|---|
{table}

## Primary holdout

| Metric | Result |
|---|---:|
| Cases | {holdout['case_count']} |
| Interventions | {primary['intervention_count']} |
| Intervention precision | {primary['intervention_precision']:.4f} |
| Precision 95% CI | [{primary['intervention_precision_ci95'][0]:.4f}, {primary['intervention_precision_ci95'][1]:.4f}] |
| Coverage | {primary['coverage']:.4f} |
| Population benefit rate | {primary['population_benefit_rate']:.4f} |
| Mean network TTT gain | {primary['mean_network_ttt_gain']:.6f} |
| Regret / safety / legal violations among interventions | {primary['regret_violation_count']} / {primary['safety_violation_count']} / {primary['legal_violation_count']} |

Engineering point targets and the stronger scientific CI condition are reported separately. Abstentions are not counted as successes or failures.

## Hypothesis outcomes

| Hypothesis | Outcome |
|---|---|
{hypothesis_table}

H13 route-overlap/WIN correlation: {holdout['statistical_tests']['H13_overlap_win_correlation']:.4f}. H14 preference-diversity × route-attribute-diversity/WIN correlation: {holdout['statistical_tests']['H14_interaction_win_correlation']:.4f}. These are descriptive synthetic associations.

## Evidence boundaries

- Microscopic phantom calibration complete: **{micro['phantom_calibration_complete']}**; separate phantom gate used: **{micro['phantom_gate_used']}**.
- Real-topology evaluation OD: `{real['evaluation_od']}`; real geometry with synthetic demand.
- Tail degradation CVaR: {tail['statistics']['H10_tail_degradation_cvar']:.6f}; frozen limit: {tail['statistics']['H10_tail_limit']:.6f}.
- TTC/PET/DRAC remain surrogate conflict indicators, never crash probabilities.
- RL remains rejected and is not used in v3.

## Final decision

**Outcome {outcome} — {outcome_text}**

**Always-on Adaptive Navigation: rejected as universal policy.**

**Selective Adaptive Navigation: Outcome {outcome}.**
"""
    OUTPUT.write_text(text, encoding="utf-8")
    payload = {
        "complete": True,
        "questions": questions,
        "hypotheses": h,
        "outcome": outcome,
        "outcome_text": outcome_text,
        "threshold_hash": _sha(frozen_path),
        "selected_model_hash": _sha(selected_path),
        "holdout_result_hash": _sha(ROOT / "artifacts/studies/v3_selective_holdout/summary.json"),
    }
    MACHINE.parent.mkdir(parents=True, exist_ok=True)
    MACHINE.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    run()
