# CONCORDIA v8 — FINAL_AUDIT_V8

## Decision

**Outcome F — the frozen deployment policy safely abstains.**

No non-empty validation point simultaneously achieved zero safety violations,
deployment precision at least 0.80, support at least 20, and unsafe recall at
least 0.95. In accordance with the preregistration, V8-F was frozen with an
impossible traffic cutoff (1.1) and zero unsafe-probability tolerance before the
new final evidence was materialized. This is an integrity-preserving failure,
not an adaptive-navigation deployment claim.

Freeze manifest self-hash:
`3e2c5b173dad223e5aa6a4a137a85871f1eca60d8f5212257aef17462f7e2ece`.

## Final audit table

| Requirement | Observed | Pass |
|---|---:|:---:|
| Development pairs ≥2,000 | 2,000 | ✅ |
| Unsafe development pairs ≥300 | 379 | ✅ |
| New microscopic final pairs ≥400 | 400 | ✅ |
| Future-state leakage | 0 | ✅ |
| Pairing failures | 0 | ✅ |
| Final top-10% traffic uplift positive | 0.00761 | ✅ |
| Safety PR-AUC reported | 0.45044 | ✅ |
| Unsafe recall ≥0.95 | 1.000 | ✅ |
| False-safe rate ≤0.05 | 0.000 | ✅ |
| Worst critical-group recall ≥0.85 | 1.000 | ✅ |
| Deployment precision ≥0.80 | 0.000 | ❌ |
| Coverage ≥0.08 | 0.000 | ❌ |
| Opportunity-realization rate ≥0.35 | 0.000 | ❌ |
| Interventions ≥30 | 0 | ❌ |
| Safety violations =0 | 0 | ✅ |
| Safe-success retention ≥0.70 | 0.000 | ❌ |
| OSM interventions >0 | 0 | ❌ |
| OSM safe interventions >0 | 0 | ❌ |
| Action-aware PR-AUC beats state-only | 0.52591 vs 0.47618 on validation | ✅ |
| Action-aware false-safe rate beats state-only | 0.00595 vs 0.01639 on validation | ✅ |
| RL decision | NO | ✅ |

Fourteen of 21 registered audit checks passed. All integrity checks passed;
the seven failed checks are deployment-effectiveness targets.

## Evidence integrity

- Development consisted of 1,500 promoted historical paired SUMO cases and 500
  newly executed paired SUMO cases. Seed-family roles were 1,200 train, 300
  calibration, and 500 validation.
- The unsafe label was fixed as
  `Risk(Adaptive) > Risk(B1) + 0.25`. No label was altered during hard-safe,
  boundary, or hard-unsafe acquisition.
- All safety inputs were available before intervention. Proposed reroute mass,
  destination load, conflict exposure, lane-change demand, bottleneck load, and
  capacity slack were expectations from the proposed action—not realized
  treatment values.
- Six frozen v8 YAML packages, development artifacts, feature/outcome/split
  definitions, and deployment code were hashed before final evaluation.
- The 400-pair microscopic final set used no development or v7-final seeds.
  The 15-OD OSM set excluded every v7 OSM OD pair.
- Freeze verification passed before and after both final evaluations.

## Final microscopic findings

The untouched 400-pair final set contained 71 SafeMicroSuccess opportunities
and 91 unsafe Adaptive counterfactuals. The traffic ranker retained a signal:
Spearman correlation was 0.407; the top 10% mean uplift was +0.761%, compared
with a population mean of −1.397%. The safety classifier achieved PR-AUC 0.450.
At the frozen zero-tolerance threshold its unsafe recall and worst critical-group
recall were both 1.0, but specificity was zero and no candidate passed.

| Policy | Interventions | Precision | Coverage | ORR | Unsafe |
|---|---:|---:|---:|---:|---:|
| B1 | 0 | 0.000 | 0.000 | 0.000 | 0 |
| B6 always-on | 400 | 0.1775 | 1.000 | 1.000 | 91 |
| V5-F | 0 | 0.000 | 0.000 | 0.000 | 0 |
| V6-Binary | 0 | 0.000 | 0.000 | 0.000 | 0 |
| V7-F | 0 | 0.000 | 0.000 | 0.000 | 0 |
| V7-Mean reconstructed | 13 | 0.5385 | 0.0325 | 0.0986 | 1 |
| V8-F | 0 | 0.000 | 0.000 | 0.000 | 0 |

## OSM transfer findings

The real-topology bridge used 15 new legal OD pairs on committed Gangnam OSM
geometry, 120 paired conditions, and 360 SUMO runs including pre-decision
probes. It contained 13 safe opportunities. V8-F and V7-F both abstained.
Always-on B6 recovered all 13 opportunities but produced 71 unsafe outcomes.
Traffic ranking did not transfer: Spearman was −0.414 and top-10% mean uplift
was negative. These are synthetic demands and preferences on real geometry,
not observed Seoul traffic.

## Hypotheses H45–H55

| Hypothesis | Status | Finding |
|---|---|---|
| H45 | PASS on validation | Calibrated unsafe filtering reduced the diagnostic false-safe intervention rate relative to v7 safety regression. |
| H46 | FAIL | Frozen V8-F was empty and did not improve precision over reconstructed V7-Mean. |
| H47 | FAIL | Non-empty final precision ≥0.80 was not achieved. |
| H48 | FAIL | Final coverage was 0. |
| H49 | PASS | Frozen final safety violations were 0. |
| H50 | FAIL | Final ORR was 0. |
| H51 | PASS at frozen threshold | Final unsafe recall was 1.0, achieved by zero-tolerance abstention. |
| H52 | FAIL | OSM had no V8-F intervention or recovered safe success. |
| H53 | PASS on validation | Action-aware features improved both PR-AUC and risk-controlled false-safe rate over state-only features. |
| H54 | PASS on microscopic final | Top-10% traffic uplift was positive and above the population mean. |
| H55 | FAIL | The filter did not reach 0.80 precision with 0.70 safe-success retention. |

## Claim boundary

V8 supports three narrow conclusions: paired traffic uplift remains rankable on
the synthetic microscopic final set; action-aware unsafe classification improves
development discrimination over state-only classification; and neither signal
is presently strong enough for a non-empty safe deployment. Safety metrics are
surrogate conflict indicators, never crash probabilities. RL remains **NO**.
