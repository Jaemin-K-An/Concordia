# CONCORDIA v10 validation decision

CONCORDIA v10 evaluated a 24-action, multi-fidelity SUMO racing policy without
reinforcement learning. The registered development process used 200 new states, at most
three repair rounds, and a separate 400-state authorization validation with mild demand
and car-following mismatch. The final holdout remained unmaterialized throughout.

## Development repairs

| Policy | Precision | Coverage | Interventions | Safety violations | Stage 1 oracle survival | Stage 2 | Stage 3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Initial | 75.86% | 14.50% | 29 | 1 | 37.84% | 19.82% | 5.41% |
| Repair 1 — candidate retention | 76.09% | 23.00% | 46 | 1 | 99.10% | 80.18% | 15.32% |
| Repair 2 — rollout fidelity | 80.65% | 15.50% | 31 | 0 | 99.10% | 81.98% | 9.91% |
| Repair 3 — robust verification | 91.30% | 11.50% | 23 | 0 | 99.10% | 81.98% | 9.91% |

Repair 1 retained all 24 statically feasible candidates through Stage 1 and increased
Stage 2 survivors to 18. Repair 2 increased Stage 2 and Stage 3 replication, raised the
recurring-unsafe evidence requirement, extended Stage 3 to 300 seconds, and increased
fresh verification to five replicas. Repair 3 required every fresh verification replica
to exceed the unchanged 0.5% benefit threshold. No success, safety, regret, or
authorization threshold was relaxed.

## Independent authorization validation

| Registered requirement | Result | Passed |
| --- | ---: | :---: |
| States = 400 | 400 | YES |
| Precision ≥ 85% | 25.53% (12/47) | NO |
| Safety violations = 0 | 9 | NO |
| Interventions ≥ 30 | 47 | YES |
| Coverage ≥ 10% | 11.75% | YES |
| Decision/evaluation seed overlap = 0 | 0 | YES |
| Final holdout materialized before decision | false | YES |

The validation policy selected 19 distinct adaptive actions. Its population mean paired
TTT gain was +0.0555%, while the mean among interventions was +0.4723%. The mutually
exclusive realized outcome classification was 12 safe-beneficial successes, 26 digital-
twin traffic false positives, and 9 safety mismatches. There were no regret or legal
violations. The p95 racing latency was 29.86 seconds and the validation consumed 101,343
recorded rollout results.

## Decision

`freeze_authorized = false`.

All three registered development repair rounds were used. The independent validation
failed both the 85% precision margin and the zero-safety-violation rule, so the v10 policy
must not be frozen and the committed 500-state final holdout must not be materialized or
run. This is a validation failure, not a final-outcome grade. RL remains `NO` and external-
topology transfer remains future work.
