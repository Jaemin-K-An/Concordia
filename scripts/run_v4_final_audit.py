#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "FINAL_AUDIT_V4.md"
MACHINE = ROOT / "artifacts/final_audit_v4.json"


def _load(relative: str):
    path = ROOT / relative
    if not path.is_file():
        raise RuntimeError(f"required v4 audit evidence is missing: {relative}")
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _status(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def run() -> Path:
    model = _load("artifacts/studies/v4_model_selection/summary.json")
    calibration = _load(
        "artifacts/studies/v4_precision_validation/calibration_summary.json"
    )
    validation = _load("artifacts/studies/v4_precision_validation/summary.json")
    validation_tests = _load(
        "artifacts/studies/v4_precision_validation/statistical_tests.json"
    )
    holdout = _load("artifacts/studies/v4_frozen_holdout/summary.json")
    micro = _load("artifacts/studies/v4_microscopic/summary.json")
    real = _load("artifacts/studies/v4_real_topology/summary.json")
    stress = _load("artifacts/studies/v4_stress/summary.json")
    manifest = _load("artifacts/v4/freeze_manifest.json")
    model_path = ROOT / "configs/v4/frozen_model.yaml"
    threshold_path = ROOT / "configs/v4/frozen_thresholds.yaml"
    primary = holdout["primary_metrics"]

    untouched = bool(
        holdout["untouched_holdout"]
        and holdout["holdout_case_ids_absent_from_development"]
    )
    frozen = bool(
        manifest["final_holdout_started"]
        and manifest["final_holdout_completed"]
        and manifest["frozen_model_hash"] == _sha(model_path)
        and manifest["frozen_threshold_hash"] == _sha(threshold_path)
        and holdout["frozen_immutable"]
    )
    questions = {
        "Holdout untouched?": _status(untouched),
        "Threshold frozen?": _status(frozen),
        "Feasibility calibrated?": _status(calibration["complete"]),
        "ECE < 0.05?": _status(calibration["selected_ece"] < 0.05),
        "Precision ≥80%?": _status(primary["intervention_precision"] >= 0.80),
        "Coverage ≥20%?": _status(primary["coverage"] >= 0.20),
        "Coverage ≥25%?": _status(primary["coverage"] >= 0.25),
        "Intervention count ≥50?": _status(primary["intervention_count"] >= 50),
        "Mean TTT gain positive?": _status(primary["mean_network_ttt_gain"] > 0.0),
        "Safety violations = 0?": _status(primary["safety_violation_count"] == 0),
        "Regret violations = 0?": _status(primary["regret_violation_count"] == 0),
        "Worst-group precision": f"{holdout['group_metrics']['worst_group_precision']:.4f}",
        "Real-topology tested?": _status(
            real["complete"] and len(real["od_pairs"]) >= 3 and real["all_routes_legal"]
        ),
        "Microscopic interventions >0?": _status(
            micro["microscopic_interventions_positive"]
        ),
        "Stress precision": f"{stress['policy_metrics']['intervention_precision']:.4f}",
        "RL used?": "NO",
    }
    hypotheses = {
        "H15 precision ≥0.80": _status(
            holdout["statistical_tests"]["H15_precision_at_least_0_80"]
        ),
        "H16 coverage >v3-D": _status(
            holdout["statistical_tests"]["H16_coverage_exceeds_v3D"]
        ),
        "H16 coverage ≥0.20": _status(
            holdout["statistical_tests"]["H16_coverage_at_least_0_20"]
        ),
        "H17 zero safety violations": _status(
            holdout["statistical_tests"]["H17_zero_safety_violations"]
        ),
        "H18 PBR >v3-D": _status(
            holdout["statistical_tests"]["H18_PBR_exceeds_v3D"]
        ),
        "H19 interaction retained": "DESCRIPTIVE — see validation CI/effect",
        "H20 ESIV improves Coverage@Precision80": _status(
            validation_tests["H20_ESIV_better"]
        ),
    }
    completion_table = "\n".join(
        f"| {question} | **{result}** |" for question, result in questions.items()
    )
    hypothesis_table = "\n".join(
        f"| {hypothesis} | **{result}** |"
        for hypothesis, result in hypotheses.items()
    )
    group_rows = []
    for dimension, groups in holdout["group_metrics"]["dimensions"].items():
        for group, metrics in groups.items():
            precision = (
                "—" if metrics["precision"] is None else f"{metrics['precision']:.4f}"
            )
            group_rows.append(
                f"| {dimension} | {group} | {metrics['intervention_count']} | "
                f"{precision} | {metrics['coverage']:.4f} |"
            )
    interaction = validation_tests["H19_interaction"]
    text = f"""# CONCORDIA final audit v4

CONCORDIA v4 separates calibrated success probability, traffic-benefit prediction, and conservative safety prediction. It selects coverage under the preregistered precision constraint, freezes every model and threshold, and then evaluates a new holdout exactly once. v2/v3 outcomes remain historical evidence and are not rewritten.

## Completion checks

| Metric | Result |
|---|---|
{completion_table}

## Primary untouched holdout

| Metric | Result |
|---|---:|
| Cases | {holdout['case_count']} |
| Selected policy | {holdout['selected_policy']} |
| Interventions / successes | {primary['intervention_count']} / {primary['successful_intervention_count']} |
| Intervention precision | {primary['intervention_precision']:.4f} |
| Precision 95% Wilson CI | [{primary['intervention_precision_ci95'][0]:.4f}, {primary['intervention_precision_ci95'][1]:.4f}] |
| Coverage | {primary['coverage']:.4f} |
| Population benefit rate | {primary['population_benefit_rate']:.4f} |
| Mean network TTT gain | {primary['mean_network_ttt_gain']:.6f} |
| Failure avoidance / missed opportunity | {primary['failure_avoidance_rate']:.4f} / {primary['missed_opportunity_rate']:.4f} |
| Regret / safety / legal violations | {primary['regret_violation_count']} / {primary['safety_violation_count']} / {primary['legal_violation_count']} |
| Strong / very strong lower-CI evidence | {holdout['statistical_tests']['strong_scientific_support']} / {holdout['statistical_tests']['very_strong_scientific_support']} |

Abstentions are excluded from intervention precision and remain in the coverage and population-benefit denominators. Engineering point targets, minimum intervention count, and scientific lower-CI conditions are reported separately.

## Hypotheses

| Hypothesis | Result |
|---|---|
{hypothesis_table}

- H19 development interaction analysis: `{json.dumps(interaction, sort_keys=True)}`.
- H20 validation Coverage@Precision80: ESIV={validation_tests['H20_ESIV_coverage_at_precision80']:.4f}, probability gate={validation_tests['H20_probability_coverage_at_precision80']:.4f}.
- Validation ECE: {validation['validation_ece']:.4f}; selected calibration method: {calibration['selected_method']}.
- Robust-CV selected model: {model['selected_model']}; worst-group precision={model['worst_group_precision']:.4f}.

## Holdout group audit

| Dimension | Group | Interventions | Precision | Coverage |
|---|---|---:|---:|---:|
{chr(10).join(group_rows)}

Median activated-group precision: **{holdout['group_metrics']['median_group_precision']:.4f}**. Worst activated-group precision: **{holdout['group_metrics']['worst_group_precision']:.4f}**. Groups with zero activation are shown with no precision rather than being counted as successful.

## External validation and boundaries

- Actual SUMO microscopic paired cases: {micro['pair_count']}; V4-F interventions: {micro['policy_metrics']['V4-F']['intervention_count']}; adaptive-success claim allowed: **{micro['adaptive_success_claim_allowed']}**.
- Real OSM-geometry OD pairs: {len(real['od_pairs'])}; all recommended paths passenger-legal: **{real['all_routes_legal']}**; demand is synthetic.
- Stress precision / coverage: {stress['policy_metrics']['intervention_precision']:.4f} / {stress['policy_metrics']['coverage']:.4f}; safety violations: {stress['policy_metrics']['safety_violation_count']}; loss CVaR: {stress['statistics']['loss_cvar']:.6f}.
- Phantom-jam prediction is secondary and not a v4 primary gate.
- TTC/PET/DRAC remain surrogate conflict indicators, never crash probabilities.
- RL remains rejected and is not part of v4.

## Final decision

**Outcome {holdout['outcome']} — {holdout['outcome_text']}**

**Always-on Adaptive Navigation remains rejected as a universal policy.**
"""
    OUTPUT.write_text(text, encoding="utf-8")
    payload = {
        "complete": True,
        "questions": questions,
        "hypotheses": hypotheses,
        "outcome": holdout["outcome"],
        "outcome_text": holdout["outcome_text"],
        "frozen_model_hash": _sha(model_path),
        "frozen_threshold_hash": _sha(threshold_path),
        "holdout_result_hash": _sha(
            ROOT / "artifacts/studies/v4_frozen_holdout/summary.json"
        ),
        "stress_result_hash": _sha(ROOT / "artifacts/studies/v4_stress/summary.json"),
        "rl_used": False,
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
