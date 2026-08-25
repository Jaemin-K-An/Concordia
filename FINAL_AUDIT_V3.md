# CONCORDIA final audit v3

The v2 negative results remain unchanged: H1, H2, H3, H4, and H6 are not rewritten by this selective-policy study. Study V is development/validation evidence only; Study VI is the primary untouched holdout.

## Completion checks

| Question | Status |
|---|---|
| Feasibility predictor calibrated? | **PARTIAL — calibration evaluated; ECE=0.117** |
| Leakage-free holdout? | **PASS** |
| Threshold frozen before holdout? | **PASS** |
| Intervention precision > 50%? | **PASS** |
| Precision engineering target ≥65%? | **FAIL** |
| Coverage ≥40%? | **FAIL** |
| Mean TTT gain positive? | **PASS** |
| Safety non-inferiority? | **PASS** |
| Real-topology selective policy tested? | **PASS** |
| Tail robustness evaluated? | **PASS** |
| RL used? | **NO** |

## Primary holdout

| Metric | Result |
|---|---:|
| Cases | 288 |
| Interventions | 15 |
| Intervention precision | 0.5333 |
| Precision 95% CI | [0.3012, 0.7519] |
| Coverage | 0.0521 |
| Population benefit rate | 0.0278 |
| Mean network TTT gain | 39.929646 |
| Regret / safety / legal violations among interventions | 0 / 0 / 0 |

Engineering point targets and the stronger scientific CI condition are reported separately. Abstentions are not counted as successes or failures.

## Hypothesis outcomes

| Hypothesis | Outcome |
|---|---|
| H8 | **FAIL** |
| H8_point | **PASS** |
| H9 | **PASS** |
| H10 | **PASS** |
| H11 | **PASS** |
| H12 | **FAIL** |
| H13 | **DESCRIPTIVE** |
| H14 | **DESCRIPTIVE** |

H13 route-overlap/WIN correlation: 0.2623. H14 preference-diversity × route-attribute-diversity/WIN correlation: 0.2044. These are descriptive synthetic associations.

## Evidence boundaries

- Microscopic phantom calibration complete: **False**; separate phantom gate used: **False**.
- Real-topology evaluation OD: `['cluster_12061673697_12061673699_4944067989', '5376448907']`; real geometry with synthetic demand.
- Tail degradation CVaR: 0.001748; frozen limit: 0.100000.
- TTC/PET/DRAC remain surrogate conflict indicators, never crash probabilities.
- RL remains rejected and is not used in v3.

## Final decision

**Outcome P — Selective CONCORDIA partially supported.**

**Always-on Adaptive Navigation: rejected as universal policy.**

**Selective Adaptive Navigation: Outcome P.**
