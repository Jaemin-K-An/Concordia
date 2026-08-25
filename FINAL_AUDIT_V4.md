# CONCORDIA final audit v4

CONCORDIA v4 separates calibrated success probability, traffic-benefit prediction, and conservative safety prediction. It selects coverage under the preregistered precision constraint, freezes every model and threshold, and then evaluates a new holdout exactly once. v2/v3 outcomes remain historical evidence and are not rewritten.

## Completion checks

| Metric | Result |
|---|---|
| Holdout untouched? | **PASS** |
| Threshold frozen? | **PASS** |
| Feasibility calibrated? | **PASS** |
| ECE < 0.05? | **FAIL** |
| Precision ≥80%? | **PASS** |
| Coverage ≥20%? | **FAIL** |
| Coverage ≥25%? | **FAIL** |
| Intervention count ≥50? | **PASS** |
| Mean TTT gain positive? | **PASS** |
| Safety violations = 0? | **PASS** |
| Regret violations = 0? | **PASS** |
| Worst-group precision | **0.4000** |
| Real-topology tested? | **PASS** |
| Microscopic interventions >0? | **PASS** |
| Stress precision | **0.4409** |
| RL used? | **NO** |

## Primary untouched holdout

| Metric | Result |
|---|---:|
| Cases | 640 |
| Selected policy | V4-P |
| Interventions / successes | 55 / 48 |
| Intervention precision | 0.8727 |
| Precision 95% Wilson CI | [0.7598, 0.9370] |
| Coverage | 0.0859 |
| Population benefit rate | 0.0750 |
| Mean network TTT gain | 84.767766 |
| Failure avoidance / missed opportunity | 0.9839 / 0.7670 |
| Regret / safety / legal violations | 0 / 0 / 0 |
| Strong / very strong lower-CI evidence | True / True |

Abstentions are excluded from intervention precision and remain in the coverage and population-benefit denominators. Engineering point targets, minimum intervention count, and scientific lower-CI conditions are reported separately.

## Hypotheses

| Hypothesis | Result |
|---|---|
| H15 precision ≥0.80 | **PASS** |
| H16 coverage >v3-D | **PASS** |
| H16 coverage ≥0.20 | **FAIL** |
| H17 zero safety violations | **PASS** |
| H18 PBR >v3-D | **PASS** |
| H19 interaction retained | **DESCRIPTIVE — see validation CI/effect** |
| H20 ESIV improves Coverage@Precision80 | **FAIL** |

- H19 development interaction analysis: `{"approximate_ci95": [-0.46139326391821667, 1.1828913902942326], "incremental_information": 0.0023993243488296123, "interaction": "heterogeneity_rad_interaction", "standardized_coefficient": 0.36074906318800803, "supports_H19": true, "validation_log_loss_full": 0.39660987202540493, "validation_log_loss_reduced": 0.39900919637423454}`.
- H20 validation Coverage@Precision80: ESIV=0.0417, probability gate=0.0417.
- Validation ECE: 0.0514; selected calibration method: isotonic.
- Robust-CV selected model: M1_logistic; worst-group precision=0.0000.

## Holdout group audit

| Dimension | Group | Interventions | Precision | Coverage |
|---|---|---:|---:|---:|
| demand_band | high | 35 | 0.8286 | 0.1367 |
| demand_band | low | 11 | 1.0000 | 0.0859 |
| demand_band | middle | 9 | 0.8889 | 0.0352 |
| heterogeneity | bimodal | 23 | 0.8261 | 0.1437 |
| heterogeneity | high | 13 | 1.0000 | 0.0813 |
| heterogeneity | long_tail | 18 | 0.8333 | 0.1125 |
| heterogeneity | low | 1 | 1.0000 | 0.0063 |
| penetration | 0.5 | 10 | 0.4000 | 0.0312 |
| penetration | 1.0 | 45 | 0.9778 | 0.1406 |
| scenario | merge | 21 | 0.9048 | 0.1313 |
| scenario | ring | 0 | — | 0.0000 |
| scenario | signalized | 18 | 0.7778 | 0.1125 |
| scenario | two_route | 16 | 0.9375 | 0.1000 |
| topology_family | constrained | 39 | 0.8462 | 0.1219 |
| topology_family | distributed | 16 | 0.9375 | 0.0500 |

Median activated-group precision: **0.8968**. Worst activated-group precision: **0.4000**. Groups with zero activation are shown with no precision rather than being counted as successful.

## External validation and boundaries

- Actual SUMO microscopic paired cases: 15; V4-F interventions / successes / surrogate safety violations: 1 / 0 / 1. The non-degenerate activation test failed, so no microscopic adaptive-success or safety-transfer claim is made.
- Real OSM-geometry OD pairs: 3; all recommended paths passenger-legal: **True**; demand is synthetic.
- Stress precision / coverage: 0.4409 / 0.1719; safety violations: 0; loss CVaR: 0.001534.
- Phantom-jam prediction is secondary and not a v4 primary gate.
- TTC/PET/DRAC remain surrogate conflict indicators, never crash probabilities.
- RL remains rejected and is not part of v4.

## Final decision

**Outcome P — High-precision CONCORDIA partially supported.**

**Always-on Adaptive Navigation remains rejected as a universal policy.**
