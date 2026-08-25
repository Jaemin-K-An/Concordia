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
6. **Offer and execution are separate state transitions.** `RouteOffer` carries truthful
   attributes and provenance; only an `ACCEPTED` `RecommendationDecision` can reach a
   simulator adapter. Rejection keeps the private/current route.
7. **Instantaneous SUMO flow is explicitly an estimate.** Edge state uses vehicle count,
   lane length, lane count, speed, and occupancy separately. When no detector count interval
   is available, flow is labelled as the traffic-state estimate \(q=k v\), not throughput.
8. **MPC executes only its first action.** Horizon assignments score anticipated feedback;
   the controller observes again before another offer. This prevents an open-loop horizon
   from masquerading as closed-loop control.
9. **RL gate Outcome A.** At the tested six-user correctness scale, no predeclared
   performance/runtime/reversal gate passed. RL was therefore not introduced. Untested
   large-scale and nonstationary behavior is a limitation, not evidence that RL is necessary.
10. **Negative results remain results.** The 2026-08-25 matrix did not support aggregate H1,
    H2, or H6. H3 was not tested by a matched microscopic policy matrix; H4 is partial.
    These outcomes are preserved in the audit instead of being tuned away.
