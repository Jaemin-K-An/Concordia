# FINAL_AUDIT_V7

Final Outcome: **Outcome F**

| Metric | Result |
|---|---:|
| Paired SUMO development ≥1000? | YES · 1200 |
| Final micro holdout ≥300? | YES · 300 |
| Pairing failures =0? | YES · development 0, final 0 |
| Future leakage =0? | YES |
| Traffic effect MAE | 0.027989 |
| Traffic effect sign accuracy | 0.523 |
| Safety effect MAE | 1.094943 |
| Deployment precision ≥80%? | NO · 0.000 |
| Coverage ≥10%? | NO · 0.000 |
| Interventions ≥30? | NO · 0 |
| ORR ≥40%? | NO · 0.000 |
| Safety violations =0? | YES · 0 |
| Precision lower CI >60%? | NO · 0.000 |
| V7 beats V6 binary selector? | NO · ORR 0.000 vs 0.000 |
| OSM interventions >0? | NO · 0 |
| OSM safe successes >0? | NO · 0 |
| Placebo test passed? | YES |
| RL used? | NO |

Final seeds and IDs are absent from fitting manifests. The five frozen YAML packages and
their source/artifact hashes were created before analytical, microscopic, or OSM final
evidence was materialized. Failed topologies and missed opportunities remain in the report.

## Reproducibility

- v6 immutable manifest: `17178971d0c52b2821a99f08bed037a3b3721c3e28a1f8cd267e37f7cd3799c3`
- v7 immutable manifest: `2a0402533fd994da80d70b7eb6561e510b4e8e97692960c6c88d6e913999ea91`
- selected treatment-effect learner: `C0_direct_paired|random_forest`
- tests: `.............                                                            [100%] / 13 passed in 1.06s`
