# CONCORDIA v8 — Safety-Filtered Uplift Ranking

## Registered design

The immutable v7 direct paired random-forest traffic ranker is followed by a calibrated, pre-decision action-aware unsafe-intervention veto. The unsafe event is `Risk(Adaptive) > Risk(B1) + 0.25`; predicted regret above 0.08 and illegal actions are also vetoed. No RL was used.

## Evidence summary

| Measure | Result |
|---|---:|
| Development paired conditions | 2000 |
| Unsafe development conditions | 379 |
| Final microscopic pairs | 400 |
| Safety model PR-AUC | 0.4504 |
| Safety model unsafe recall | 1.0000 |
| V8-F interventions | 0 |
| V8-F precision | 0.0000 |
| V8-F coverage | 0.0000 |
| V8-F ORR | 0.0000 |
| V8-F safety violations | 0 |
| Safe-success retention | 0.0000 |
| OSM paired conditions | 120 |
| OSM V8-F interventions | 0 |
| OSM V8-F safe successes | 0 |

## Validation selection

The registered lexicographic search evaluated 150 policy points and found 0 feasible points. Safe abstention was `True`.

## State vs action-aware safety

State-only validation PR-AUC was 0.4762; the selected action-aware model achieved 0.5259. This comparison is reported as measured, without promoting a failed hypothesis.

## Traffic ranking

On the untouched final set, the population mean paired traffic uplift was -0.0140; the top 10% mean was 0.0076, with Spearman correlation 0.4072.

## Transfer and limits

The OSM bridge used 15 newly selected Gangnam OD pairs, 120 paired conditions, and synthetic demand. It is evidence on real geometry, not observed Seoul traffic. Failure taxonomy and mechanism-proxy plots are in `artifacts/studies/v8_failure_analysis/`.

## Reproducibility

Freeze manifest self-hash: `3e2c5b173dad223e5aa6a4a137a85871f1eca60d8f5212257aef17462f7e2ece`. The final microscopic and OSM evaluations verified the same frozen hash before and after evaluation.
