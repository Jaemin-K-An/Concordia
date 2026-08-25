<p align="center">
  <img src="assets/brand/concordia-lockup.svg" width="750" alt="Concordia" />
</p>

<p align="center">
  <strong>Truthful, preference-aligned, safety-constrained adaptive navigation.</strong><br />
  A reproducible research system for voluntary route coordination—not vehicle control.
</p>

<p align="center">
  <img alt="Python 3.9+" src="https://img.shields.io/badge/Python-3.9%2B-111111?style=flat-square" />
  <img alt="SUMO 1.27.1" src="https://img.shields.io/badge/SUMO-1.27.1-404040?style=flat-square" />
  <img alt="Tests 44 passing" src="https://img.shields.io/badge/tests-44%20passing-111111?style=flat-square" />
  <img alt="RL gate Outcome A" src="https://img.shields.io/badge/RL%20gate-Outcome%20A-737373?style=flat-square" />
</p>

<p align="center">
  <a href="#system">System</a> ·
  <a href="#architecture">Architecture</a> ·
  <a href="#reproduce">Reproduce</a> ·
  <a href="#research-status">Research status</a> ·
  <a href="#evidence">Evidence</a>
</p>

---

> [!IMPORTANT]
> CONCORDIA recommends legal routes using truthful computed attributes. It never controls
> speed, steering, acceleration, or lane changes. A route is changed only after the modeled
> user explicitly accepts the offer.

## System

CONCORDIA asks whether heterogeneous private preferences can become a coordination resource:
can the network reduce congestion externalities by finding routes that some users genuinely
prefer—or can accept with very little regret—without forcing sacrifice or degrading safety?

| Behavioral layer | Adaptive layer | Research layer |
|---|---|---|
| Independent route choice and acceptance | Dynamic state and horizon prediction | B0–B6 matched policy comparisons |
| Population prior and user posterior | Multi-objective candidate generation | Paired seeds, bootstrap CI, effect sizes |
| Pairwise learning and preference drift | Acceptance-aware receding-horizon MPC | SUMO/SSM and real OSM verification |
| Accepted-only route execution | Exact, greedy, HiGHS MIP, and MPC baselines | Mandatory quantitative RL gate |

### What is implemented

- Explicit `RouteOffer` → `RecommendationDecision` → `AcceptanceOutcome` domain stages.
- Count, density, flow, speed, and occupancy state with documented units and provenance.
- ETA, reliability, cost, risk, complexity, and familiarity route candidates with overlap and
  Pareto filtering.
- Dynamic route attributes and Preference Slack over a projected traffic horizon.
- Closed-loop MPC that offers only its first action, observes again, and never executes a
  rejected recommendation.
- Separate phantom-jam risk predictor and multi-event trajectory detector.
- TTC, PET, DRAC, hard-braking, SSM conflict typing, and direction-aware safety-tail metrics.
- Original-geometry OSM import, topology audit, SUMO conversion, and QGIS-ready GeoJSON.
- Immutable run manifests containing commit/dirty state, timing, versions, hardware, seeds,
  validity, and input/output hashes.

## Architecture

```mermaid
flowchart LR
    A["SUMO / analytical network"] --> B["Network state estimator"]
    B --> C["Risk + route prediction"]
    C --> D["Preference posterior"]
    D --> E["Constrained MPC optimizer"]
    E --> F["Truthful route offer"]
    F --> G{"User accepts?"}
    G -->|"yes"| H["Execute route change"]
    G -->|"no"| I["Keep current/private route"]
    H --> A
    I --> A
```

The analytical layer is a BPR correctness harness. The microscopic layer is actual SUMO
1.27.1 through TraCI. They are never silently substituted for one another.

## Reproduce

```bash
make setup
make lint
make test
make benchmark
make experiment
make research
make simulation-test
make phantom-calibrate
make alignment-study
make microscopic-study
make real-topology-study
make scalability-study
make drift-study
make rl-gate
make rl-evaluate
make report-v2
make audit-v2
```

Install the official SUMO/TraCI optional dependencies before microscopic validation:

```bash
python3 -m pip install -e '.[dev,analysis,sumo]'
```

The fast test suite does not open SUMO. CI separates analytical, microscopic, and manually
dispatched research-matrix workflows.

## Research status

The committed study contains **240 screening rows** and **60 focused paired B0–B6 rows**.
These are synthetic analytical correctness-scale experiments, not observed traffic results.

| Hypothesis | Status | Finding |
|---|---|---|
| H1 · Preference diversity creates more diversion opportunities | **FAIL** | Not supported by the declared synthetic population and routes. |
| H2 · B6 reduces TTT under bounded regret | **FAIL** | B6 respected the regret bound but did not beat B1 in aggregate. |
| H3 · Adaptive routing prevents phantom jams | **FAIL** | In 60 matched pairs, VALID-event probability was B1=0.033 and B6=0.100 (exact McNemar p=0.289). |
| H4 · Safety tails do not degrade | **FAIL** | DRAC-CVaR non-inferiority was not established: B6−B1 mean 0.183, upper 95% bootstrap bound 0.464 > 0.25 margin. |
| H5 · Variance/tails explain more than means | **PARTIAL** | Exploratory in-sample association only. |
| H6 · Closed-loop feedback is more stable | **FAIL** | No pre-registered stability effect was met. |
| H7 · RL exceeds constrained MPC | **NOT TESTED** | The mandatory gate did not authorize RL. |

The signalized scenario exposes an important failure condition: ETA-only can lower network TTT
while exceeding some users' regret budget. CONCORDIA reports that trade-off instead of hiding
it behind an aggregate objective.

### Microscopic and real-topology boundary

- The paired SUMO matrix completed 120 runs and 60 B1/B6 pairs; all 11,960 generated vehicles
  arrived. Only physically VALID events count toward H3.
- Of 45 event candidates, 8 met the three-detector physical validation contract. Predictor
  calibration remains **FAILED / NOT CALIBRATED** because the held-out seed did not contain
  both VALID-event classes.
- The Gangnam-area OSM experiment completed 20 B1/B6 runs on three legal alternatives and
  1,000/1,000 arrivals. B6 increased mean transfer TTT by about 3.8%; its OD remains explicitly
  **synthetic demand on real topology**.
- Surrogate conflicts are never interpreted as crash probabilities.

### RL decision

> **Outcome A — RL not introduced. B6 enumeration reached a declared scale limit, but the
> hard-constrained clustered approximation met the frozen latency/quality thresholds; median
> Gate C degradation also stayed below its threshold.**

This outcome does not claim that RL can never help. The preference-drift p95 remained a recorded
failure tail even though the pre-registered median Gate C did not trigger.

## Evidence

<table>
  <tr>
    <td width="50%"><img src="artifacts/figures/ttt_vs_demand.png" alt="TTT versus demand" /></td>
    <td width="50%"><img src="artifacts/figures/phase_diagram.png" alt="Beneficial-diversion phase diagram" /></td>
  </tr>
  <tr>
    <td align="center"><sub>Synthetic analytical TTT across B0–B6</sub></td>
    <td align="center"><sub>Screened opportunity surface—not a real-world effect map</sub></td>
  </tr>
</table>

| Artifact | Contents |
|---|---|
| [`FINAL_AUDIT_V2.md`](FINAL_AUDIT_V2.md) | Final completion checks and evidence-based hypothesis outcomes |
| [`artifacts/reports/final_report_v2.html`](artifacts/reports/final_report_v2.html) | Alignment, microscopic, real-topology, scalability, and RL paper-style report |
| [`alignment_frontier/summary.json`](artifacts/studies/alignment_frontier/summary.json) | Price of Alignment, knee uncertainty, and WIN/TRADEOFF/INFEASIBLE map |
| [`microscopic_policy_matrix/summary.json`](artifacts/studies/microscopic_policy_matrix/summary.json) | 120 actual SUMO runs, H3 paired events, and H4 DRAC-CVaR non-inferiority |
| [`phantom_calibration/summary.json`](artifacts/studies/phantom_calibration/summary.json) | Leakage-safe held-out calibration attempt and explicit failure reason |
| [`real_topology_policy_matrix/summary.json`](artifacts/studies/real_topology_policy_matrix/summary.json) | Actual B1/B6 SUMO transfer on Gangnam geometry with synthetic demand |
| [`scalability/summary.json`](artifacts/studies/scalability/summary.json) | Enumeration boundary, approximation runtime/memory, and Gate E |
| [`preference_drift/summary.json`](artifacts/studies/preference_drift/summary.json) | Nonstationarity performance, tail behavior, and Gate C |
| [`artifacts/rl_gate_report_v2.json`](artifacts/rl_gate_report_v2.json) | Frozen Gate C/E reevaluation and Outcome A |
| [`analytical_matrix/summary.json`](artifacts/studies/analytical_matrix/summary.json) | Screening, focused B0–B6 rows, paired statistics, and failure cases |
| [`analytical_matrix/manifest.json`](artifacts/studies/analytical_matrix/manifest.json) | Clean source commit, runtime, dependencies, and hashes |
| [`sumo_ring/summary.json`](artifacts/studies/sumo_ring/summary.json) | Microscopic traffic, wave candidates, safety distributions, and claim boundary |
| [`sumo_ring/manifest.json`](artifacts/studies/sumo_ring/manifest.json) | Clean source commit, SUMO version, inputs, runtime, and output hashes |
| [`real_topology/topology_audit.json`](artifacts/studies/real_topology/topology_audit.json) | OSM topology, components, route diversity, and demand provenance |
| [`docs/mathematical_spec.md`](docs/mathematical_spec.md) | Units, utility, constraints, MPC, safety, and scientific assumptions |

## Repository map

```text
assets/        Figma-derived Concordia brand assets
configs/       scenario, population, policy, screening, and focused experiment inputs
data/          immutable OSM source, processed geometry, and provenance manifests
docs/          mathematical specification, decisions, gap audit, and RL gate decision
src/concordia/ behavior, adaptive loop, optimization, simulation, traffic, safety, GIS
tests/         unit, property-style, integration, regression, and golden scenarios
experiments/   candidate factor-space declaration
scripts/       SUMO and real-topology verification helpers
artifacts/     phase reports, study evidence, figures, registry, gate, and report
```

---

<p align="center">
  <sub>Safety metrics are surrogate conflict indicators, never accident probabilities.<br />
  Synthetic preference and acceptance coefficients are assumptions, not calibrated human behavior.</sub>
</p>
