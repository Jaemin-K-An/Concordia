# CONCORDIA v10 racing engine

The v10 engine implements the preregistered 24→12→6→3→1 successive-halving race.
Stage 0 removes only illegal, preference-infeasible, zero-acceptance, structurally
over-capacity, or invalid-route actions. Stage 1 uses short paired SUMO signals and a
fixed uncertainty band; Stage 2 uses replicated mean-minus-variance scoring; Stage 3
requires a positive empirical Q10 benefit with zero unsafe replicas; three fresh
verification replicas confirm the finalists. B1 remains available throughout as a
fallback and is never treated as a racing candidate.

Every request receives a deterministic content-addressable seed. Decision seeds are
checked against realized evaluation seeds, collisions are resolved deterministically
inside the registered range, and final realized outcomes are prohibited from the
rollout cache. The simulator records truncated vehicle-time, queue, bottleneck-load,
DRAC risk, regret, legality, acceptance, network hash, and route-file hash for every
fidelity.

The engine is independent of a traffic surrogate. Existing v9 models can later break
ties or allocate extra rollout budget, but the implementation contains no path that
discards an action solely because of predicted rank.
