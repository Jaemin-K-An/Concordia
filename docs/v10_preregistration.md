# CONCORDIA v10 preregistration

CONCORDIA v10 tests whether preference-feasible adaptive actions can be selected by
successive, multi-fidelity microscopic SUMO racing rather than by a one-shot static
surrogate ranking. The v9 action library is retained: 24 adaptive actions and the
always-available B1 null action. Surrogates may allocate budget or break ties, but may
not independently eliminate an action.

The registered default race is:

1. Stage 0 removes only structurally impossible or preference-infeasible actions.
2. Stage 1 evaluates all valid actions for 60 seconds with one replica and retains 12
   using conservative uncertainty-aware elimination.
3. Stage 2 evaluates survivors for 120 seconds with three independent replicas and
   retains six using mean benefit minus 0.5 standard deviations plus safety checks.
4. Stage 3 evaluates six actions for 240 seconds with five replicas, requires a Q10
   benefit above 0.005, zero unsafe replicas, risk delta at most 0.25, regret at most
   0.08, and legal routes, then retains three.
5. Three fresh verification replicas recheck the finalists. The eligible action with
   the largest Stage-3 Q10 is selected; otherwise B1 is executed.

Development uses 200 new states. A separate 400-state validation partition authorizes
freeze only when precision is at least 0.85, coverage at least 0.10, interventions at
least 30, and safety violations equal zero. At most three registered repair rounds are
allowed, without changing benefit, safety, regret, or precision criteria.

The v9 final seed family remains permanently unused historical reservation. A new
500-state v10 final seed manifest may be materialized only after an authorized frozen
commit is present on remote `main`. At preregistration time no v10 final state, outcome,
or seed manifest exists.

Primary final success requires precision at least 0.80, coverage at least 0.10, at
least 40 interventions, and zero safety violations. RL is fixed to `NO`; OSM transfer
is secondary and cannot change the primary microscopic outcome.

