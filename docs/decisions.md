# Research decisions

1. **Analytical core before SUMO.** Mathematical invariants and exact small-instance checks
   must pass before microscopic claims are attempted.
2. **Concentration uses HHI in the objective.** Entropy is an excellent diagnostic but its sign
   makes a minimization objective easy to misuse; HHI is minimized and entropy is reported.
3. **Hard safety filtering.** Safety non-degradation is checked before objective comparison.
   A penalty cannot buy permission to violate the configured bound.
4. **No silent simulator fallback.** Analytical runs are explicitly labelled. Requesting SUMO
   without a valid executable raises an environment error.
5. **Exact oracle is the correctness reference.** Greedy and future learning policies are
   compared against exhaustive enumeration on tractable instances.
