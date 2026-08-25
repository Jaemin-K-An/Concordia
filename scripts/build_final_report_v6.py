#!/usr/bin/env python3
from __future__ import annotations

import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "artifacts/reports"


def _read(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text())


def _outcome(micro: dict, osm: dict) -> str:
    metrics = micro["primary_metrics"]
    if metrics["safety_violation_count"] > 0 or metrics["precision"] < 0.60:
        return "F"
    if (
        metrics["precision"] >= 0.80
        and metrics["coverage"] >= 0.15
        and metrics["opportunity_recovery_rate"] >= 0.50
        and metrics["intervention_count"] >= 50
        and osm["primary_metrics"]["success_count"] > 0
    ):
        return "S+"
    if (
        metrics["precision"] >= 0.80
        and metrics["coverage"] >= 0.10
        and metrics["opportunity_recovery_rate"] >= 0.40
        and metrics["intervention_count"] >= 30
    ):
        return "S"
    return "P" if metrics["precision"] >= 0.60 else "F"


def run() -> Path:
    dataset = _read("artifacts/studies/v6_micro_dataset/dataset_summary.json")
    validation = _read("artifacts/studies/v6_policy_validation/validation_summary.json")
    candidates = _read("artifacts/studies/v6_micro_model_selection/candidate_comparison.json")
    ablations = _read("artifacts/studies/v6_policy_validation/ablation.json")
    analytical = _read("artifacts/studies/v6_frozen_analytical_holdout/summary.json")
    micro = _read("artifacts/studies/v6_frozen_micro_holdout/summary.json")
    osm = _read("artifacts/studies/v6_real_topology/summary.json")
    failure = _read("artifacts/studies/v6_failure_analysis/summary.json")
    primary = micro["primary_metrics"]
    outcome = _outcome(micro, osm)
    selected = validation["selected"]
    model_rows = "\n".join(
        f"| {row['name']} | {row['strategy']} | {row['selected_calibration']} | "
        f"{row['validation_metrics']['auc']:.3f} | {row['validation_metrics']['average_precision']:.3f} | "
        f"{row['validation_metrics']['ece']:.3f} |"
        for row in candidates
    )
    ablation_rows = "\n".join(
        f"| {row['ablation']} | {row.get('auc', float('nan')):.3f} | "
        f"{row.get('average_precision', float('nan')):.3f} | "
        f"{row['metrics']['precision']:.3f} | {row['metrics']['coverage']:.3f} |"
        for row in ablations
    )
    report = f"""# CONCORDIA v6 Final Research Report

## Executive result

The preregistered v6 pipeline was completed with **Outcome {outcome}**. The development study used {dataset['pair_count']} paired actual-SUMO cases ({dataset['actual_sumo_run_count']} runs), followed by a frozen, seed-disjoint {micro['pair_count']}-pair microscopic holdout. Validation found no non-empty operating point satisfying precision ≥0.80 and zero safety violations; the frozen policy therefore used safe abstention and left B1 unchanged. This is a negative deployment result, not evidence of adaptive-routing success.

## Research progression

CONCORDIA began with preference-aligned adaptive routing, rejected always-on deployment after microscopic mismatch, and progressed through selective analytical gating, calibration, domain-shift detection, and v5 micro correction. v6 replaced that stack with a direct pre-decision predictor of `SafeMicroSuccess`, defined jointly by ≥1% TTT benefit, DRAC-CVaR margin ≤0.25, affected-user regret ≤0.08, and route legality.

## Data and leakage controls

- Development: {dataset['pair_count']} paired cases; SafeMicroSuccess {dataset['safe_micro_success_count']} ({dataset['safe_micro_success_rate']:.1%}).
- Split: train 360, calibration 120, validation 120 by disjoint seed family.
- Final microscopic holdout: {micro['pair_count']} new paired cases / {micro['actual_sumo_run_count']} SUMO runs.
- Pairing failures: development {dataset['pairing_failure_count']}; final {micro['pairing_failure_count']}.
- Future-state leakage: {dataset['future_state_leakage_count']} detected cases.
- RL was not used because the residual problem remained classification and domain transfer.

## Model and calibration comparison

| Model | Scope | Calibration | Validation ROC AUC | Validation AP | ECE |
|---|---|---:|---:|---:|---:|
{model_rows}

The precision-constrained validation frontier selected `{selected['method']}` with model `{selected['model']}` and architecture `{selected['architecture']}`. Thresholds were frozen before any analytical, microscopic, or OSM final result was materialized.

## Final microscopic evidence

| Metric | V6-F result |
|---|---:|
| Interventions | {primary['intervention_count']} |
| Precision | {primary['precision']:.3f} |
| 95% Wilson lower bound | {primary['precision_wilson_95_lower']:.3f} |
| Coverage | {primary['coverage']:.3f} |
| Opportunity Recovery Rate | {primary['opportunity_recovery_rate']:.3f} |
| Safety violations | {primary['safety_violation_count']} |
| False-safe rate | {primary['false_safe_rate']:.3f} |
| Inference p95 | {micro['predictor_inference_p95_seconds']:.6f} s |

The analytical precision-preserving frozen reference achieved precision {analytical['reference_metrics']['intervention_precision']:.3f}; the separate recall-oriented stage-1 screen recovered {analytical['v6_stage1_opportunity_recall']:.1%} of analytical opportunities and was not treated as a deployment claim.

## Ablation

| Ablation | AUC | AP | Policy precision | Coverage |
|---|---:|---:|---:|---:|
{ablation_rows}

These comparisons do not rescue the deployment claim: no ablation may override the frozen validation constraint or use final outcomes for threshold tuning.

## Real OSM geometry

The transfer study used {osm['od_pair_count']} stratified OD pairs and {osm['paired_condition_count']} paired conditions on committed Gangnam OSM geometry with synthetic demand/preferences. V6-F made {osm['primary_metrics']['intervention_count']} interventions and found {osm['primary_metrics']['success_count']} safe successes. This is **real road geometry with synthetic OD demand**, not an estimate of observed Seoul traffic effects.

## Failure mechanisms and boundary

Failure analysis visualized {failure['visualized_case_count']} cases with full paired flow, speed, queue, DRAC/TTC proxy, route-load, and x–t speed/density figures. Its basis was `{failure['analysis_basis']}` and thresholds were not changed. TTC, PET, and DRAC are surrogate conflict indicators, never crash probabilities.

## Hypotheses H29–H36

- H29: evaluated against V5-F on the same microscopic holdout; see policy metrics artifact.
- H30: {'supported' if primary['precision'] >= 0.80 and primary['intervention_count'] > 0 else 'not supported'}.
- H31: {'supported' if primary['opportunity_recovery_rate'] >= 0.40 else 'not supported'}.
- H32: {'supported' if primary['safety_violation_count'] == 0 else 'not supported'}.
- H33/H34: evaluated by temporal and analytical feature ablations; no final-data model revision was permitted.
- H35: penetration-stratified development rates are reported as effect-modification evidence, not causality.
- H36: {'supported' if osm['primary_metrics']['success_count'] > 0 else 'not supported'}.

## Conclusion

Outcome {outcome} means the evidence does not support a strong adaptive-navigation deployment claim under the preregistered criteria. The defensible engineering output is the frozen safe-abstention policy, the reproducible paired-SUMO dataset, and a precise account of where micro-domain classification and topology transfer remain unresolved.
"""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    markdown = REPORT_DIR / "CONCORDIA_V6_FINAL_REPORT.md"
    markdown.write_text(report, encoding="utf-8")
    (REPORT_DIR / "CONCORDIA_V6_FINAL_REPORT.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>CONCORDIA v6</title>"
        "<style>body{max-width:1000px;margin:40px auto;font:16px/1.55 system-ui;color:#17212b}"
        "pre{white-space:pre-wrap}</style><pre>" + html.escape(report) + "</pre>",
        encoding="utf-8",
    )
    print(markdown)
    return markdown


if __name__ == "__main__":
    run()
