# CONCORDIA v7 Final Research Report

## Result

**Outcome F**

CONCORDIA v7 changed the primary question from binary success prediction to paired conditional
treatment-effect estimation. The frozen policy intervenes only when the traffic-effect lower
bound exceeds 1%, the safety-effect upper bound is at most 0.25, predicted maximum regret is at
most 0.08, and the route is legal. SafeMicroSuccess is retained only as the deployment precision
label. No final microscopic or OSM result was used to tune a model, interval, or threshold.

The development corpus contains **1200 paired cases**, of which
**400** are newly generated v7 actual-SUMO cases. The final
microscopic holdout contains **300 fresh paired cases / 600
SUMO runs**. V7-F made **0 interventions**, with deployment precision
**0.000**, coverage **0.000**, ORR
**0.000**, and **0** safety
violations. The 95% Wilson precision lower bound is **0.000**.

## Ten required research questions

1. **Why was binary classification insufficient?** v6 collapses benefit magnitude, safety effect,
   and regret into one label. On the same v7 final holdout V6-Binary recovered
   0.000 of safe opportunities, while the
   continuous model ranked traffic effects at Spearman
   0.482 and separately exposed poor safety-effect
   transfer. That magnitude information was scientifically useful, but V7-F also recovered zero
   opportunities: replacing the binary target did not by itself solve safe deployment.

2. **How predictable was the continuous paired effect?** Final PEHE-like traffic-effect MAE was
   **0.0280**, RMSE **0.0451**,
   and Spearman correlation **0.482**. These are direct
   errors against paired counterfactual effects, not observational causal proxies.

3. **How often was the traffic-uplift sign correct?** Final sign accuracy was
   **0.523** and positive-uplift recall was
   **0.215**.

4. **How stable was safety-effect prediction?** Final safety-effect MAE was
   **1.0949**, RMSE **3.0876**,
   and the all-case false-safe rate under the frozen safety UCB was
   **0.097**.

5. **Did interval selection improve precision over mean selection?** On the identical final
   holdout, V7-mean precision was **0.654**,
   bootstrap-quantile precision **0.000**, and
   conformal precision **0.000**. H38 is reported
   strictly from these frozen comparisons; zero-intervention variants are not credited with
   perfect precision.

6. **Did treatment-effect selection improve ORR over v6?** V7-F ORR was
   **0.000** versus V6-Binary
   **0.000** on the same final cases.

7. **What did high navigation penetration do?** The table below reports paired effects rather
   than assuming monotonic improvement. At 100% penetration the observed mean effect was
   **-0.0026**; comparison with
   lower penetration is descriptive within the synthetic randomized design.

8. **How did effects differ by topology?** The topology table shows substantial heterogeneity;
   it is the evidence used for this answer, including unsuccessful topologies rather than
   filtering them after the fact.

9. **Was positive uplift identified on real OSM geometry?** The frozen policy made
   **0** interventions and recovered
   **0** safe beneficial cases across
   **12** prespecified Gangnam OD pairs, despite
   **17** paired counterfactual safe opportunities.
   Demand and preferences remain
   synthetic; this is not an observed-Seoul causal claim.

10. **Where was Adaptive structurally negative?** The condition analysis found
    **8** descriptive axis levels with negative mean paired effect.
    They were `acceptance=0.6, acceptance=0.8, acceptance=1.0, demand_band=high, penetration=0.5, penetration=0.75, penetration=1.0, topology=signalized`. These regimes are preserved in the failure artifact;
    plausible mechanisms include secondary
    bottlenecks, partial adoption mismatch, and topology transfer, but the mechanism labels are
    diagnostics rather than separately identified causal effects.

## Final policy comparison

| Policy | Interventions | Precision | Coverage | ORR | Safety violations |
|---|---:|---:|---:|---:|---:|
| B6 always-on | 300 | 0.223 | 1.000 | 1.000 | 48 |
| V5-F | 0 | 0.000 | 0.000 | 0.000 | 0 |
| V6-Binary | 0 | 0.000 | 0.000 | 0.000 | 0 |
| V7-T | 0 | 0.000 | 0.000 | 0.000 | 0 |
| V7-TS | 0 | 0.000 | 0.000 | 0.000 | 0 |
| V7-TSU | 0 | 0.000 | 0.000 | 0.000 | 0 |
| **V7-F** | **0** | **0.000** | **0.000** | **0.000** | **0** |

## Effect by penetration

| Penetration | N | Mean effect | Median effect | Positive ≥1% |
|---:|---:|---:|---:|---:|
| 0.25 | 75 | 0.0071 | 0.0000 | 0.280 |
| 0.50 | 75 | -0.0093 | 0.0024 | 0.373 |
| 0.75 | 75 | -0.0269 | 0.0000 | 0.160 |
| 1.00 | 75 | -0.0026 | 0.0000 | 0.240 |

## Effect by topology

| Topology | N | Mean effect | Median effect | Positive ≥1% |
|---|---:|---:|---:|---:|
| asymmetric | 60 | 0.0171 | 0.0080 | 0.483 |
| merge | 60 | 0.0009 | 0.0041 | 0.450 |
| real_like | 60 | 0.0049 | 0.0000 | 0.117 |
| signalized | 60 | -0.0664 | -0.0627 | 0.083 |
| two_route | 60 | 0.0039 | 0.0000 | 0.183 |

## Hypotheses H37–H44

| Hypothesis | Status |
|---|---:|
| H37 | **FAIL** |
| H38 | **FAIL** |
| H39 | **FAIL** |
| H40 | **FAIL** |
| H41 | **FAIL** |
| H42 | **FAIL** |
| H43 | **PASS** |
| H44 | **FAIL** |

H37 uses Coverage@Precision80 on the same final holdout. H38 compares frozen mean and quantile
variants. H39 compares V7-T with V7-TS safety violations. H40 compares ORR with V6-Binary on the
same cases. H41–H44 apply the preregistered final thresholds literally.

## Robustness and diagnostics

- Development validation selected `C0_direct_paired|random_forest` and
  `bootstrap_quantile` uncertainty handling.
- The paired placebo real-signal Spearman was
  0.459, versus
  0.010 after target permutation.
- Three development-only family holdouts were executed: leave_asymmetric_topology_out, leave_high_demand_band_out, leave_full_penetration_band_out.
- Historical analytical V5-RD precision on the 512-case correctness check was
  0.833.
- Failure analysis retained 0 false positives and
  67 missed safe opportunities without topology exclusion.

## Claim boundary

The paired treatment effects are identified inside randomized synthetic SUMO experiments. The
OSM study preserves real road geometry but uses synthetic OD demand and modeled preferences.
TTC, PET, and DRAC are surrogate conflict indicators, not crash outcomes. CONCORDIA changes only
legal route recommendations after modeled acceptance; it does not control vehicle motion. RL was
not used because the v7 question is conditional-effect estimation and conservative selection.
