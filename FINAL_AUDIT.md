# Final audit status

This file separates verified properties of the current analytical release from claims that
require later empirical phases.

| Audit question | Current evidence/status |
|---|---|
| Is individual utility actually used? | Yes. Exact and greedy policies calculate normalized utility and reject routes beyond per-user regret budgets. |
| Does the policy use false information? | No. Explanations and utility use computed route features only; no incident-message generator exists. |
| Is efficiency bought with safety degradation? | The optimizer hard-filters aggregate route-risk exposure against the baseline plus delta. Microscopic TTC/PET non-inferiority is not yet established. |
| Is heterogeneity the cause of improvement? | Mechanism-ready paired synthetic populations exist; a full statistical matrix is still required for a causal research claim. |
| Is RL necessary? | Not demonstrated; therefore RL is deliberately not a final policy in this release. |
| When can RL beat the optimizer? | Unknown. Future tests must compare unseen demand/topology under matched constraints. |
| Is “phantom jam” dynamically evidenced? | Not yet. The analytical ghost score is labelled a risk proxy; wave claims require SUMO trajectories. |
| Are surrogate risks distinguished from crash risk? | Yes, in the mathematical spec, report, and metric names. |
| Is statistical uncertainty shown? | Bootstrap confidence intervals, standard deviation, tails, and paired effect size are implemented. Full-matrix evidence is pending. |
| Is same-config reproduction supported? | Yes for deterministic analytical runs; config, seeds, versions, and commit are recorded. |
| Does the finding hold on a real GIS network? | Not tested. No real-world claim is made. |
| Are failure conditions stated? | Non-convergence, infeasible constraints, missing SUMO, malformed OD/routes, and unverified real-world transfer are explicit. |

## Definition-of-done assessment

- `make setup`, `make test`, `make benchmark`, `make experiment`, and `make report` are wired.
- The deterministic analytical core, golden scenarios, safety parser, exact oracle, greedy
  baseline, bandit learner, registry, and report builder are implemented and tested.
- SUMO/QGIS real-road calibration, full SSM event export, large-scale batch results, learned
  ghost-risk calibration, interactive map UI, and any justified constrained-RL/GNN policy are
  open research phases. The repository does not present them as completed findings.
