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

## Required audit checklist

| Metric | Result |
|---|---|
| New untouched analytical holdout? | YES — 1024 cases |
| New untouched microscopic holdout? | YES — 100 pairs |
| Freeze immutable? | YES |
| Analytical precision ≥80%? | YES — 0.8258 |
| Analytical coverage ≥15%? | NO — 0.1289 |
| Analytical coverage ≥20%? | NO — 0.1289 |
| Overall lower CI >70%? | YES — 0.7521 |
| Critical-group precision ≥70%? | YES — 0.7400 |
| Stress precision ≥70%? | YES — 0.8519 |
| Micro interventions ≥10? | YES — 10 |
| Micro successful interventions >0? | YES — 1 |
| Micro safety violations =0? | NO — 1 |
| False-safe rate ≤5%? | NO — 0.1000 |
| Real OSM intervention >0? | NO — 0 |
| Calibration ECE <0.05? | NO — 0.05093 |
| RL used? | NO |

## H21–H28

- **H21_regime_conditioning: PASS** — Frozen holdout V5-R slightly improved both precision and coverage over V5-G.
- **H22_DSS: FAIL** — DSS retained zero safety violations but did not improve stress precision over the no-DSS ablation.
- **H23_micro_safety_gate: FAIL / OVER-CONSERVATIVE** — The safety gate reduced analytical activation to zero and still allowed one microscopic false-safe intervention.
- **H24_micro_correction: FAIL** — Microscopic benefit correction slightly worsened final-holdout MAE.
- **H25_safe_micro_success: FAIL** — There was one safe success, but precision was 0.10 and one safety violation occurred.
- **H26_selectivity_mechanism: PARTIAL** — Selectivity removed all analytical safety violations and 24/25 microscopic unsafe adaptations, but not the last false-safe case.
- **H27_hierarchical_or_mixture: FAIL** — Model selection chose regime-specific M3, not the hierarchical or mixture candidates.
- **H28_penetration_interactions: FAIL** — The interaction model did not jointly improve validation coverage and Brier score over its nested global comparator.

## Answers to the ten final questions

1. v4 used a global synthetic-domain gate. Its historical 50%-penetration precision was 0.40, stress precision was 0.4409, and its only SUMO intervention was unsafe. The missing regime and domain bridge allowed synthetic calibration to be mistaken for transfer.
2. Penetration is a dominant activation variable, but not a sufficient microscopic success explanation. In v5 analytical holdout, p=1.0 produced 128/132 interventions at 0.8516 precision; p=0.25 and p=0.50 produced none. In SUMO, p=1.0 produced 7/10 selected interventions but only one success and one unsafe case.
3. Only slightly on analytical holdout: V5-R precision/coverage were 0.8309/0.1328 versus global 0.8271/0.1299. This small gain did not transfer to full SUMO.
4. Weakly descriptive, not operationally supported. Mild-shift SUMO cases had lower raw success (0.10 versus 0.222 in-distribution), but DSS did not improve stress precision over the no-DSS ablation; H22 fails.
5. Across all 100 SUMO pairs, mean analytical probability minus realized success was +0.0083. Selection bias was more important: the ten selected cases had mean corrected-benefit overprediction +0.0218, yielding 0.10 realized precision.
6. Mostly, but not completely. Always-adapt B6 had 25 unsafe cases; the selective policy reduced this to one among ten interventions. Because zero was required, the veto fails.
7. Not with the frozen v5 policy. It retained 0.8258 precision but reached only 0.1289 coverage.
8. Yes for the analytical stress domain: precision was 0.8519, coverage 0.1055, safety violations zero.
9. No. All 48 conditions across six legal stratified OSM OD pairs abstained.
10. The frozen policy permanently falls back for strong shift, LOW_CONTROL and PARTIAL_CONTROL analytical cells without a validated threshold, illegal route sets, nonpositive corrected micro benefit, low micro success probability, or a micro safety UCB veto.


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
