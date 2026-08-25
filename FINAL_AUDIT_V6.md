# FINAL_AUDIT_V6

Final Outcome: **F**

| Metric | Result | Value |
|---|---:|---:|
| New microscopic dataset ≥500? | YES | 600 |
| Final micro holdout untouched? | YES | 200 disjoint pairs |
| Micro precision ≥80%? | NO | 0.000 |
| Micro coverage ≥10%? | NO | 0.000 |
| Micro interventions ≥30? | NO | 0 |
| Opportunity Recovery ≥40%? | NO | 0.000 |
| Micro safety violations =0? | YES | 0 |
| False-safe rate ≤5%? | YES | 0.000 |
| Precision lower CI >60%? | NO | 0.000 |
| Analytical precision ≥80%? | YES | 0.847 |
| Real OSM interventions >0? | NO | 0 |
| Real OSM safe success >0? | NO | 0 |
| Temporal features useful? | YES | development validation ablation |
| Analytical score useful in micro model? | YES | development validation ablation |
| RL used? | NO | classification/domain-transfer problem |

The outcome is computed from frozen final evidence. No final outcome was used for model, calibration, architecture, conformal cutoff, or threshold selection.

## Reproducibility

- v5 frozen source commit preserved: `15a8723fdfe6cfa06583441ea18d89fcb6de1b86`
- v6 frozen source commit: `456893fa4eb2e1057eabc4910b0ebfda06c631b8`
- v6 unit tests: `.........                                                                [100%] / 9 passed in 0.75s`
- validation selection: `safe_abstention_when_validation_constraints_infeasible`
