# CONCORDIA v9 optimizer validation

The preregistered ranking repair was executed after the initial surrogate failed the
Top-5 candidate-recall criterion. Repair Round 2 added pre-decision state-action
interactions, a within-state pairwise ranker, a calibration-selected candidate
ensemble, and the registered Top-7 development diagnostic.

The repaired screen reached 0.8125 Top-5 oracle recall on exhaustive calibration
states, but did not reproduce on the isolated exhaustive validation states:

- validation states: **24**
- repaired Top-5 oracle recall: **0.375** (Gate B requires 0.80)
- repaired Top-7 diagnostic recall: **0.4583**
- pairwise-only Top-5 recall: **0.5417**
- selected safety model validation unsafe-class average precision: **0.5922**

This is a gate-preserving negative result. Gate C digital-twin discrimination and
Gate D deployment feasibility were not reached, no policy was frozen, and no final
holdout state was materialized. Repair Round 3 is not eligible because its registered
trigger requires a successful Top-M screen followed by rollout/safety failure.

The result therefore does not authorize the remaining freeze, holdout, or final-audit
commits. This prevents calibration overfit from being converted into a deployment
claim and preserves all 500 registered final states as untouched.
