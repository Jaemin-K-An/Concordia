# CONCORDIA v5 Final Audit

## Decision

**Outcome F.** Final microscopic safety failure forces Outcome F despite analytical precision and stress precision passing.

The deployment code and five frozen YAML packages match the pre-holdout SHA-256 manifest. No
final seed entered model fitting, regime discovery, shift fitting, calibration, micro correction,
safety-veto fitting, or threshold selection.

## Primary evidence

| Domain | N / interventions | Precision | Coverage | Safety violations | Decision |
|---|---:|---:|---:|---:|---|
| Analytical | 1024 / 132 | 0.8258 | 0.1289 | 0 | precision pass; coverage fail |
| Stress | 256 / 27 | 0.8519 | 0.1055 | 0 | stress target pass |
| Actual SUMO microscopic | 100 / 10 | 0.1000 | 0.1000 | 1 | claim forbidden |
| Real OSM geometry | 48 / 0 | 0.0000 | 0.0000 | 0 | all abstain |

Analytical precision passed 0.80 with 132 interventions and zero analytical safety violations,
but coverage was 0.1289 rather than 0.15. Stress precision was 0.8519. In actual SUMO, the
full policy made 10 interventions, achieved one success, and allowed one surrogate safety
violation; false-safe rate was 0.10. That microscopic safety failure independently forces F.

## H21–H28

- **H21_regime_conditioning: PASS** — Frozen holdout V5-R slightly improved both precision and coverage over V5-G.
- **H22_DSS: FAIL** — DSS retained zero safety violations but did not improve stress precision over the no-DSS ablation.
- **H23_micro_safety_gate: FAIL / OVER-CONSERVATIVE** — The safety gate reduced analytical activation to zero and still allowed one microscopic false-safe intervention.
- **H24_micro_correction: FAIL** — Microscopic benefit correction slightly worsened final-holdout MAE.
- **H25_safe_micro_success: FAIL** — There was one safe success, but precision was 0.10 and one safety violation occurred.
- **H26_selectivity_mechanism: PARTIAL** — Selectivity removed all analytical safety violations and 24/25 microscopic unsafe adaptations, but not the last false-safe case.
- **H27_hierarchical_or_mixture: FAIL** — Model selection chose regime-specific M3, not the hierarchical or mixture candidates.
- **H28_penetration_interactions: FAIL** — The interaction model did not jointly improve validation coverage and Brier score over its nested global comparator.


## Transparent aggregation correction

The frozen microscopic evaluator incorrectly included B6 TTT deltas for abstained rows when
computing its descriptive population mean. The stored decisions, intervention count, precision,
success labels, and safety labels are unaffected. Recomputed from immutable raw pairs, the
selected policy's population-mean network TTT gain is
**1.7900 s**
(relative **0.000370**).
The original value remains preserved in the raw summary and no post-freeze threshold changed.

## Claim boundary

- Analytical evidence is synthetic and uses a BPR correctness harness.
- Microscopic evidence is actual SUMO with synthetic demand and preferences.
- OSM supplies real geometry only; its OD demand is synthetic.
- TTC, PET, and DRAC are surrogate conflict indicators, never crash probabilities.
- RL remained excluded because the v5 gate did not authorize it.
