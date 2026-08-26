# CONCORDIA v9 preregistration

V9 changes the decision problem from accepting or rejecting one B6 action to
choosing the best action from a preference-feasible state-specific library. An
action specifies a user subset, route allocation, diversion intensity, and
allocation probabilities. The null action is always present and executes B1.

The initial library contains 25 actions per state (null plus 24 deterministic
balanced combinations) spanning all six registered diversion intensities,
eight user-selection strategies, and six route-allocation strategies. Every
selected user must satisfy Preference Slack at most its epsilon before the
action is created. Regret at most 0.08 and route legality remain hard checks.

Development uses 500 seed-family-defined states. One hundred stratified states
receive exhaustive actual-SUMO action evaluation; the remaining states receive
a registered broad actual-action subset after all candidates are generated.
Oracle Actionability Rate is therefore reported both on the exhaustive stratum
and as an all-state evaluated-action lower bound.

The final 500-state seed list is declared but must not be materialized until:

1. development OAR is at least 0.40;
2. Top-5 oracle recall is at least 0.80;
3. digital-twin safe-benefit precision is at least 0.70;
4. a validation policy makes at least 20 interventions at precision at least
   0.80 with zero safety violations; and
5. six deployment YAML files and the v9 manifest have been frozen.

At most three development-only repair rounds are allowed. Their triggers and
permitted changes are fixed in `configs/v9/repair_protocol.yaml`. Final evidence
may never alter the action library, thresholds, safety margin, regret limit, or
seed list. OSM is a secondary transfer study. RL is not authorized.

