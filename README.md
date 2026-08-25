# Concordia Adaptive Navigation

Concordia is a reproducible research system for **truthful, preference-aligned,
safety-constrained route recommendation**. It does not control speed, steering, acceleration,
or lane changes. It observes a traffic network, estimates dynamic route attributes, proposes a
route, models the user's independent acceptance/rejection, and changes a simulator route only
after acceptance.

The repository now contains:

- explicit `RouteOffer`, `RecommendationDecision`, and `AcceptanceOutcome` domain stages;
- behavioral softmax choice, a provenance-labelled logistic acceptance model, population
  priors, online user posteriors, pairwise preference learning, and preference forgetting;
- count/density/flow/speed/occupancy state with explicit units and an analytical/SUMO boundary;
- multi-objective ETA/reliability/cost/risk/complexity candidates with deduplication, overlap,
  and Pareto filtering;
- dynamic horizon route prediction and Preference Slack;
- a closed-loop acceptance-aware MPC controller that executes only its first accepted action;
- B0–B6 baselines: static, ETA-only, preference-only, SO oracle, greedy VDE, HiGHS MIP, and MPC;
- separate phantom-jam predictor/detector APIs, calibration metrics, SUMO SSM parsing, TTC,
  PET, DRAC, hard-braking and safety-tail non-degradation checks;
- a real OSM topology/geometry/provenance pipeline and QGIS-compatible GeoJSON;
- screening and focused paired-seed experiments, immutable run manifests, automatic figures,
  a mandatory RL gate, and a hypothesis-indexed audit.

## Reproduce

```bash
make setup
make lint
make test
make benchmark
make experiment
make research
make simulation-test
make rl-gate
make rl-evaluate
make report
make audit
```

`make simulation-test` needs the official SUMO/TraCI packages from the `sumo` optional extra:

```bash
python3 -m pip install -e '.[dev,analysis,sumo]'
```

The fast test suite does not open SUMO. CI separates the analytical and microscopic jobs.

## Current results — read before citing

The committed B0–B6 study is a **synthetic analytical correctness-scale experiment**, not an
observed traffic result. Its 240 screening rows and 60 focused paired rows did not support
aggregate H1 or H2; H5 is exploratory and H6 is not supported by a pre-registered stability
criterion. The signalized case exposes a useful failure mode: ETA-only may reduce TTT while
violating some users' regret budget, whereas the constrained policies preserve the budget.

The SUMO 1.27.1 ring smoke run produced complete trajectories, explicit traffic state, SSM
output, safety distributions, and two detector candidates for backward-propagating waves.
Because there is no matched adaptive-vs-ETA microscopic probability matrix, **H3 phantom-jam
prevention is NOT TESTED**. Microscopic safety non-degradation H4 is **PARTIAL**, not a crash
claim. The real Gangnam-area OSM extract preserves geometry and converts successfully with
`netconvert`; its OD demand is explicitly **synthetic demand on real topology**, and the same
mechanism has not yet been demonstrated there.

The mandatory quantitative gate produced **Outcome A**:

> RL not introduced because deterministic/receding-horizon optimization was sufficient under
> the tested, declared small-instance conditions.

This does not claim that RL can never help. Larger networks and full nonstationary traffic
studies remain untested, and absence of evidence is not treated as a gate trigger.

## Evidence map

- [`docs/mathematical_spec.md`](docs/mathematical_spec.md) — units, objectives, constraints,
  acceptance-aware MPC, and claim boundary.
- [`docs/implementation_gap_audit.md`](docs/implementation_gap_audit.md) — baseline audit made
  before implementation.
- [`artifacts/studies/analytical_matrix/summary.json`](artifacts/studies/analytical_matrix/summary.json)
  — screening/focused B0–B6 rows and paired hypothesis statistics.
- [`artifacts/studies/sumo_ring/summary.json`](artifacts/studies/sumo_ring/summary.json) — actual
  microscopic smoke-run version, hashes, wave candidates, and safety summary.
- [`artifacts/studies/real_topology/topology_audit.json`](artifacts/studies/real_topology/topology_audit.json)
  and [`sumo_conversion.json`](artifacts/studies/real_topology/sumo_conversion.json) — OSM and
  SUMO topology verification.
- [`artifacts/rl_gate_report.json`](artifacts/rl_gate_report.json) and
  [`docs/rl_gate_decision.md`](docs/rl_gate_decision.md) — RL thresholds and Outcome A.
- [`artifacts/reports/report.html`](artifacts/reports/report.html) — generated research report.
- [`FINAL_AUDIT.md`](FINAL_AUDIT.md) — PASS/PARTIAL/NOT TESTED status by claim.

## Repository map

```text
configs/       scenario, population, policy, screening, and focused experiment inputs
data/          immutable OSM source, processed geometry, and provenance manifests
docs/          mathematical specification, decisions, gap audit, and RL gate decision
src/concordia/ behavior, adaptive loop, optimization, simulation, traffic, safety, GIS
tests/         unit, property-style, integration, regression, and golden scenarios
experiments/   candidate factor-space declaration
scripts/       SUMO and real-topology verification helpers
artifacts/     phase reports, study evidence, figures, registry, gate, and report
```

Safety metrics throughout this project are surrogate conflict indicators, never accident
probabilities. Synthetic preference/acceptance coefficients are labelled as assumptions and
must not be presented as calibrated human behavior.
