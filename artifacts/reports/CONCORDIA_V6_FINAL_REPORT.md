# CONCORDIA v6 Final Research Report

## Executive result

The preregistered v6 pipeline was completed with **Outcome F**. The development study used 600 paired actual-SUMO cases (1200 runs), followed by a frozen, seed-disjoint 200-pair microscopic holdout. Validation found no non-empty operating point satisfying precision ≥0.80 and zero safety violations; the frozen policy therefore used safe abstention and left B1 unchanged. This is a negative deployment result, not evidence of adaptive-routing success.

## Research progression

CONCORDIA began with preference-aligned adaptive routing, rejected always-on deployment after microscopic mismatch, and progressed through selective analytical gating, calibration, domain-shift detection, and v5 micro correction. v6 replaced that stack with a direct pre-decision predictor of `SafeMicroSuccess`, defined jointly by ≥1% TTT benefit, DRAC-CVaR margin ≤0.25, affected-user regret ≤0.08, and route legality.

## Data and leakage controls

- Development: 600 paired cases; SafeMicroSuccess 101 (16.8%).
- Split: train 360, calibration 120, validation 120 by disjoint seed family.
- Final microscopic holdout: 200 new paired cases / 400 SUMO runs.
- Pairing failures: development 0; final 0.
- Future-state leakage: 0 detected cases.
- RL was not used because the residual problem remained classification and domain transfer.

## Model and calibration comparison

| Model | Scope | Calibration | Validation ROC AUC | Validation AP | ECE |
|---|---|---:|---:|---:|---:|
| M0_logistic | global | isotonic | 0.667 | 0.428 | 0.025 |
| M1_interaction_logistic | global | isotonic | 0.688 | 0.431 | 0.028 |
| M2_gradient_boosting | global | isotonic | 0.664 | 0.332 | 0.069 |
| M3_random_forest | global | isotonic | 0.680 | 0.406 | 0.072 |
| M4_calibrated_gradient_boosting | global | isotonic | 0.640 | 0.325 | 0.075 |
| MR_regime_boosting | regime | isotonic | 0.705 | 0.434 | 0.023 |
| MH_hierarchical_logistic | hierarchical | isotonic | 0.678 | 0.437 | 0.044 |

The precision-constrained validation frontier selected `safe_abstention_when_validation_constraints_infeasible` with model `MH_hierarchical_logistic` and architecture `C_composite_plus_safety_veto`. Thresholds were frozen before any analytical, microscopic, or OSM final result was materialized.

## Final microscopic evidence

| Metric | V6-F result |
|---|---:|
| Interventions | 0 |
| Precision | 0.000 |
| 95% Wilson lower bound | 0.000 |
| Coverage | 0.000 |
| Opportunity Recovery Rate | 0.000 |
| Safety violations | 0 |
| False-safe rate | 0.000 |
| Inference p95 | 0.001338 s |

The analytical precision-preserving frozen reference achieved precision 0.847; the separate recall-oriented stage-1 screen recovered 100.0% of analytical opportunities and was not treated as a deployment claim.

## Ablation

| Ablation | AUC | AP | Policy precision | Coverage |
|---|---:|---:|---:|---:|
| without_traffic_temporal | 0.664 | 0.429 | 0.000 | 0.000 |
| without_analytical | 0.677 | 0.435 | 0.000 | 0.000 |
| without_topology | 0.640 | 0.415 | 0.000 | 0.000 |
| without_preference | 0.692 | 0.424 | 0.000 | 0.000 |
| without_penetration | 0.671 | 0.434 | 0.000 | 0.000 |
| without_safety | 0.664 | 0.426 | 0.000 | 0.000 |
| no_stage1_screener | nan | nan | 0.000 | 0.000 |
| no_safety_veto | nan | nan | 0.000 | 0.000 |

These comparisons do not rescue the deployment claim: no ablation may override the frozen validation constraint or use final outcomes for threshold tuning.

## Real OSM geometry

The transfer study used 10 stratified OD pairs and 80 paired conditions on committed Gangnam OSM geometry with synthetic demand/preferences. V6-F made 0 interventions and found 0 safe successes. This is **real road geometry with synthetic OD demand**, not an estimate of observed Seoul traffic effects.

## Failure mechanisms and boundary

Failure analysis visualized 3 cases with full paired flow, speed, queue, DRAC/TTC proxy, route-load, and x–t speed/density figures. Its basis was `highest_score_counterfactual_B6_failures_under_safe_abstention` and thresholds were not changed. TTC, PET, and DRAC are surrogate conflict indicators, never crash probabilities.

## Hypotheses H29–H36

- H29: evaluated against V5-F on the same microscopic holdout; see policy metrics artifact.
- H30: not supported.
- H31: not supported.
- H32: supported.
- H33/H34: evaluated by temporal and analytical feature ablations; no final-data model revision was permitted.
- H35: penetration-stratified development rates are reported as effect-modification evidence, not causality.
- H36: not supported.

## Conclusion

Outcome F means the evidence does not support a strong adaptive-navigation deployment claim under the preregistered criteria. The defensible engineering output is the frozen safe-abstention policy, the reproducible paired-SUMO dataset, and a precise account of where micro-domain classification and topology transfer remain unresolved.
