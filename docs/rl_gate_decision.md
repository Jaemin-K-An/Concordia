# RL gate decision

## Outcome A

**RL not introduced because deterministic/receding-horizon optimization was sufficient under the tested, declared small-instance conditions.**

No quantitatively demonstrated unsolved problem passed a gate. Untested dynamic generalization and large-scale behavior are limitations, not evidence for RL.

| Gate | Tested | Triggered | Evidence |
|---|---:|---:|---|
| A_performance_gap | yes | no | >5% median B6 disadvantage against regret-feasible B4 in at least 2 scenario families |
| B_runtime | yes | no | >5 seconds in more than 20% of focused recommendation cycles |
| C_dynamic_generalization | no | no | no trained policy exists and drift evidence is unit-level, not a full traffic study |
| D_feedback_instability | yes | no | accepted one-way route changes are not counted as feedback oscillation |
| E_scalability | no | no | focused study is a 6-user correctness scale; large-scale solver evidence is absent |

## Claim boundary

This decision applies only to the tested analytical correctness scale and does not claim that RL can never help on larger or nonstationary networks.

The machine-readable evidence is `artifacts/rl_gate_report.json`.
