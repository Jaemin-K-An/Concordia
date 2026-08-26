# CONCORDIA v8 preregistration

This phase replaces continuous safety-effect prediction with a pre-decision,
action-aware classifier for the event `Risk(Adaptive) > Risk(B1) + 0.25`.
The v7 direct paired random-forest traffic uplift learner remains the primary
ranker. A candidate is ranked for benefit first, then vetoed by calibrated
unsafe probability, predicted regret above 0.08, or legal/execution checks.

The development set contains 1,500 promoted v7 pairs plus 500 newly executed
paired SUMO conditions. Seed families are split into train/calibration/
validation partitions. A completely new 400-pair microscopic final holdout and
a 120-condition OSM transfer set remain inaccessible until all six deployment
YAML files and the v8 freeze manifest exist.

Policy selection is lexicographic: zero validation safety violations, precision
at least 0.80, support at least 20, maximum opportunity-realization rate, then
maximum coverage. If no candidate is feasible, the registered deployment is
safe abstention. No RL experiment is authorized in v8.

All classifier inputs are available before treatment. Action quantities are
expected exposure and load deltas computed from the proposed reroute and the
pre-decision state; realized acceptance, realized post-treatment flow, and all
counterfactual outcomes are forbidden features.
