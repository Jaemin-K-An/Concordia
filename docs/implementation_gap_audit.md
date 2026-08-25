# CONCORDIA implementation gap audit

Audit date: 2026-08-25 (Asia/Seoul)  
Audited HEAD: `ead504e675af00e9aff2d3eeba49f748f2cc1ba7`

The repository was inspected before modification. The baseline commands `make lint`,
`make test`, and `make benchmark` passed (24 tests). This document distinguishes software
presence from scientific validation; an implemented API is not automatically evidence for a
research claim.

## Implemented

- Normalized private utility, softmax choice probabilities, Preference Slack, and regret.
- Diverse simple-path generation with overlap and Pareto filtering.
- BPR road network, Frank-Wolfe Wardrop UE and SO, PoA, and flow-conservation tests.
- Exact small-population assignment oracle and greedy VDE heuristic with hard regret, route
  legality, and aggregate route-risk constraints.
- Synthetic two-route, Braess, ring, merge, and signalized analytical graph fixtures.
- Trajectory-frame TTC/DRAC/hard-braking summaries and a SUMO SSM XML parser.
- An analytical ghost-risk logistic proxy and a sustained backward-wave event detector.
- Synthetic mean-matched preference populations and a LinUCB baseline.
- Reproducible YAML smoke config, paired seeds, bootstrap summaries, decision JSONL records,
  GeoJSON export, run registry, HTML summary report, and fast CI.

## Partially implemented

- SUMO adapter and ring input files exist, but the adapter does not distinguish edge count,
  density, flow, speed, and occupancy correctly and no microscopic run artifact exists.
- Recommendation explanations are numeric, but recommendation, user decision, and execution
  are not separate domain stages.
- Safety constraints use static route-risk exposure. SSM results are parsed but not connected
  to the experiment loop, and non-inferiority does not yet cover all distribution tails.
- Candidate paths are dynamically evaluated only through BPR assignment; generation itself is
  free-flow-time focused rather than multi-scalarized.
- The run registry records commit and dependencies, but not dirty state, end time, hardware,
  solver versions, input/output hashes, or invalid-run status.
- The report is a smoke-run table, not a hypothesis-indexed research report with figures.

## Incorrect or misleading

- `SumoAdapter.step()` assigns `getLastStepVehicleNumber()` to both density and flow. Vehicle
  count is dimensionless; density and flow require length/lane/time conversions or detectors.
- `SimulationAdapter.recommend_route()` maps directly to SUMO `setRoute`, so an offer can be
  executed without an explicit acceptance outcome.
- The README calls the current code a mesoscopic harness although no mesoscopic state-evolution
  loop exists. The code is an analytical static harness plus unvalidated SUMO fixtures.
- The analytical `GhostRiskModel` coefficients are uncalibrated and cannot support a phantom-jam
  prevention claim.

## Missing

- RouteOffer/decision/acceptance types, calibrated-vs-synthetic coefficient provenance, user
  rejection behavior, and execution guards.
- Dynamic state estimator, route-attribute predictor, dynamic Preference Slack, and a single
  closed-loop orchestrator.
- Receding-horizon baseline, stochastic constrained assignment, and a scalable MIP baseline.
- Predictor/event separation with predictor calibration metrics and multiple tracked events.
- Population prior, user posterior, pairwise/dueling learner, and preference-drift evaluation.
- Screening/focused experiment design; B0–B6 matched-policy comparison; H1–H7 primary outcomes;
  ablations, counterfactuals, computational scaling, and nonparametric paired statistics.
- Real OSM import with original geometry, topology checks, SUMO conversion, and real-topology
  synthetic-demand verification.
- SUMO CI, simulation smoke artifacts, phase reports, RL gate report/decision, and explicit RL
  outcome A/B/C.

## Scientifically unvalidated

- H1–H7 are not yet tested by a predeclared matched design.
- No microscopic evidence establishes phantom-jam event probability, duration, wave speed,
  affected length, or prevention.
- No SSM-based safety non-degradation result exists.
- No actual user-choice data calibrates utility or acceptance parameters.
- No real-topology mechanism replication exists.
- No evidence yet establishes that RL is necessary or beneficial.

## Blocked by external data or runtime

- SUMO and TraCI were absent in the baseline environment; microscopic validation must remain
  invalid until a real simulator run produces complete trajectory and SSM artifacts.
- No observed OD matrix, crash data, trajectory data, survey, or route-choice dataset is
  supplied. Any real topology must therefore be labelled **synthetic demand on real topology**.
- Network retrieval requires a licensed OSM extract and a checksum-bearing manifest.

## Future-only until its gate passes

- PPO, GNN-PPO, constrained RL, and MARL. RL must not be implemented unless the quantitative
  gate identifies an unresolved performance, runtime, nonstationarity, feedback-stability, or
  scalability problem.

## Immediate acceptance criteria

1. Correct and unit-test SUMO edge quantities and units.
2. Make recommendation → acceptance → accepted execution an enforced domain boundary.
3. Add dynamic state/prediction and a deterministic closed-loop analytical smoke test.
4. Add receding-horizon and solver-explicit MIP baselines before any RL work.
5. Generate phase reports and an RL gate whose default result is non-introduction when evidence
   is absent.

