# CONCORDIA v9 development actionability

This development-only study evaluates a preference-feasible library of 24 adaptive
actions plus the always-available null action. It uses 500 newly generated SUMO
states, including 100 exhaustive-oracle states, and preserves common pre-decision
state, network, route file, and random numbers within every counterfactual action set.

The initial library passes Gate A without an action-space repair:

- oracle actionability rate: **0.458** (threshold 0.40)
- exhaustive-oracle actionability rate: **0.580** across 100 states
- single-B6 safe-success rate: **0.176**
- multi-action improvement over B6: **+0.282**
- states with an interior optimum among evaluated actions: **163**
- paired SUMO runs: **6,600**, with zero pairing failures

All labels require relative traffic benefit above 0.005, safety delta at most 0.25,
maximum affected regret at most 0.08, and legal execution. The 500-state final
holdout remains unmaterialized.

Three state-action traffic surrogates are developed on seed-family training data and
compared on exhaustive calibration states: random forest, gradient boosting, and a
histogram-quantized gradient booster. Calibration Top-5 oracle recall is below Gate B,
so no policy is frozen here. The preregistered round-2 ranking repair is deferred to
the validation/optimizer commit.
