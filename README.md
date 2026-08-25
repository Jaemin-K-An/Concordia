# Concordia Adaptive Navigation

Concordia is a reproducible research harness for **preference-aligned, safety-constrained
route recommendation**. It never controls speed, lane changes, or vehicle actuators. Its
only intervention is a truthful route recommendation that respects a per-user utility-loss
budget.

The first research release implements the deterministic mathematical core required before
microscopic simulation or reinforcement learning:

- Yen-style diverse candidate routes with overlap and Pareto filtering;
- normalized private utility, bounded-rational choice, Preference Slack, and user regret;
- Wardrop user equilibrium (UE), system optimum (SO), and Price of Anarchy on BPR networks;
- exact small-network assignment oracle and a greedy marginal-benefit baseline;
- hard route-feasibility, user-regret, and aggregate safety constraints;
- two-route, Braess, ring-road, merge, and signalized golden scenarios;
- trajectory-derived TTC, PET, DRAC, hard-braking, and tail-risk summaries;
- a deterministic experiment registry that records config, seed, versions, and Git commit;
- an optional SUMO/TraCI adapter that fails fast when SUMO is unavailable.

## Quick start

```bash
make setup
make test
make benchmark
make experiment
make report
```

The core test command uses the Python standard library and does not require SUMO. The
experiment writes an immutable run directory under `artifacts/runs/`; the report is rebuilt
from those recorded metrics.

## Reproduce a run

```bash
PYTHONPATH=src python3 -m concordia.cli experiment \
  --config configs/experiments/smoke.yaml
```

Every config explicitly declares its seed. Randomness is split into Python and NumPy streams,
and the registry records the current code commit. See [the mathematical specification](docs/mathematical_spec.md)
for units, assumptions, objectives, and limitations.

## Scope and scientific claims

This repository distinguishes implemented evidence from planned work. Synthetic results are
never described as observed traffic. Safety measures are surrogate conflict indicators, not
crash probabilities. The current release is a validated analytical and mesoscopic harness;
large SUMO experiment matrices, calibrated real-road demand, QGIS exports, GNN policies, and
constrained RL remain later research phases and must not be claimed without data.

## Repository map

```text
configs/       versioned scenario, population, policy, and experiment inputs
data/          immutable raw-data boundary, processed outputs, provenance manifests
docs/          notation, mathematical specification, and research decisions
src/concordia/ domain code, independent from simulator APIs
tests/         unit, property-style, integration, regression, and scenario tests
experiments/   experiment-matrix definitions
scripts/       environment and external-simulator helpers
artifacts/     run registry and generated reports (outputs ignored by Git)
```
