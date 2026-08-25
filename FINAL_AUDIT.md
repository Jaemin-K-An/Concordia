# CONCORDIA final audit

Audit date: 2026-08-25 (Asia/Seoul)<br>
Implementation base HEAD: `ead504e675af00e9aff2d3eeba49f748f2cc1ba7`<br>
Final working-tree commit: recorded after this audit is committed

Statuses mean exactly: **PASS** = directly verified by committed evidence; **FAIL** = tested
and the proposed hypothesis/criterion was not met; **PARTIAL** = some required evidence exists
but the full criterion is not established; **NOT TESTED** = no valid experiment supports a
decision. An implemented API alone never upgrades a scientific claim.

## System and research audit

| Question | Status | Evidence and boundary |
|---|---|---|
| Explicit SUMO count/density/flow/speed/occupancy semantics | **PASS** | `src/concordia/simulation/base.py`, `src/concordia/simulation/sumo.py`, `tests/test_behavior_and_sumo_state.py`; flow is explicitly the instantaneous \(q=kv\) estimate, not detector throughput. |
| Recommendation separated from user choice and execution | **PASS** | `src/concordia/behavior/`, `src/concordia/adaptive/controller.py`; rejection tests prove zero `setRoute` calls. |
| Truthful route-offer schema with model provenance | **PASS** | `RouteOffer` contains ETA, variance, cost, safety, complexity, familiarity, utility, slack, marginal benefit, acceptance, time, version, and coefficient source. Coefficients are labelled synthetic. |
| Dynamic state, route features, and Preference Slack | **PASS** | `src/concordia/adaptive/state.py`, `prediction.py`; projection-sensitive slack regression test passes. |
| Observe–plan–offer–accept/reject–execute–reobserve loop | **PASS** | `ClosedLoopController`; analytical regression verifies first-action execution and replanning. |
| Exact, greedy, MIP, and receding-horizon layers are distinct | **PASS** | `src/concordia/optimization/`; B5 identifies HiGHS linearization, B6 identifies enumerative MPC correctness scale, and no silent fallback exists. |
| Multi-objective route candidates are genuinely distinct | **PASS** | ETA/reliability/cost/risk/complexity scalarizations, duplicate/overlap/Pareto filters, and route-diversity tests. |
| Preference prior, posterior, pairwise learner, and drift mechanism | **PARTIAL** | Unit evidence verifies all four APIs; no traffic-level drift comparison exists. |
| Actual SUMO microscopic execution | **PASS** | [`artifacts/studies/sumo_ring/summary.json`](artifacts/studies/sumo_ring/summary.json): SUMO 1.27.1, 6,000 steps, 37,802 trajectory frames, FCD/SSM/output hashes. Synthetic ring only. |
| Phantom predictor/detector separation | **PASS** | `src/concordia/traffic/phantom.py` and `waves.py`; multiple-event and calibration-metric tests pass. |
| SUMO-calibrated phantom predictor | **NOT TESTED** | No multi-demand/perturbation train/test dataset. Logistic/tree APIs and unit calibration data are not final calibration evidence. |
| H3 phantom-jam prevention | **NOT TESTED** | One smoke run has two detector candidates, but no matched ETA-only vs adaptive event-probability matrix. No prevention claim is made. |
| SSM and trajectory safety pipeline | **PASS** | SSM file is complete and parsed; TTC/DRAC/hard-braking distributions and CVaR are stored in [`safety_distributions.json`](artifacts/studies/sumo_ring/safety_distributions.json). SSM threshold conflict count happened to be zero. |
| H4 safety non-degradation | **PARTIAL** | Tail non-inferiority logic is tested, but no matched microscopic baseline/proposed policy pairs establish non-degradation. Metrics are surrogate conflicts, not crash probability. |
| Real OSM geometry/provenance/topology | **PASS** | [`data/manifests/gangnam_intersection.json`](data/manifests/gangnam_intersection.json), original geometry GeoJSON, 278 nodes, 427 edges, one weak component, three alternatives. |
| Real OSM SUMO conversion | **PASS** | [`sumo_conversion.json`](artifacts/studies/real_topology/sumo_conversion.json): netconvert 1.27.1, deterministic output hash, 835 edge elements. |
| Same mechanism on real topology | **NOT TESTED** | No traffic simulation was executed on the converted network. OD is explicitly **synthetic demand on real topology**. |
| Run registry completeness and invalid-run exclusion | **PASS** | Manifests record config, commit/dirty state, Python/dependencies/SUMO/solver/RL versions, timestamps, hardware, seeds, input/output hashes, validity, and reasons. Reports read only valid runs. |
| Separate analytical and SUMO CI | **PASS** | `.github/workflows/ci.yml` has `analytical-core` and `simulation` jobs; the latter installs official SUMO wheels and executes ring plus real conversion. |
| Automated report and figures | **PASS** | [`artifacts/reports/report.html`](artifacts/reports/report.html), 11 generated figures, QGIS-compatible congestion/risk layer. NOT TESTED plots are labelled rather than synthesized. |

## Hypothesis audit

Primary evidence is [`artifacts/studies/analytical_matrix/summary.json`](artifacts/studies/analytical_matrix/summary.json):
240 screening rows and 60 focused rows, paired over seeds 11/23/37/53/71.

| Hypothesis | Status | Result |
|---|---|---|
| H1 — more heterogeneity creates more low-regret beneficial diversion | **FAIL** | High/long-tail mean opportunities 2.092 versus low/none 4.000. The declared synthetic population and route features did not support H1. |
| H2 — TTT reduction under near-No-Sacrifice | **FAIL** | Aggregate paired B1−B6 mean improvement was −751.231 vehicle-min/hour; bootstrap 95% CI [−1072.263, −447.772]. B6 respected \(\epsilon=0.08\), but did not beat B1 overall. |
| H3 — adaptive lowers phantom-jam event probability | **NOT TESTED** | Missing matched multi-seed microscopic policy study. |
| H4 — efficiency does not worsen safety tail risk | **PARTIAL** | Constraint code and one SUMO safety distribution exist; policy-level paired non-inferiority is absent. |
| H5 — variance/tails explain more than means | **PARTIAL** | In-sample variance/tail \(R^2=0.0720\) versus mean-weight \(R^2=0.0012\); exploratory only, no held-out or causal evidence. |
| H6 — closed-loop is more stable than open-loop | **FAIL** | Descriptive CVs are close (B4 0.3952, B6 0.3945), but no pre-registered validated stability effect was met. Route-reversal gate remained below threshold. |
| H7 — RL beats MPC in specific dynamic conditions | **NOT TESTED** | RL gate Outcome A did not authorize implementation; therefore no RL performance claim exists. |

## Policy and computational findings

- B0–B6 all execute in the analytical matrix. B3 is the small-system oracle; B5 is a
  current-state linearization and visibly fails on some congestion cases, which is reported.
- B6 median disadvantage to regret-feasible B4 was below the 5% RL gate in every family:
  merge −1.82%, ring +2.58%, signalized 0%, two-route +0.80%.
- B6 end-to-end p95 latency was about 3.3 seconds and maximum about 3.5 seconds at six users;
  this passed the declared five-second correctness-scale gate, but is not large-scale evidence.
- Signalized ETA-only achieved lower TTT while reaching maximum regret 0.1505, above the
  configured 0.08. This is a concrete failure mode: preserving private utility can prevent the
  traffic-optimal diversion, and the trade-off must not be hidden.
- Full component ablations, large-scale memory curves, and microscopic B0–B6 matrices remain
  **PARTIAL/NOT TESTED**.

## RL final decision

**Outcome A — RL not introduced because deterministic/receding-horizon optimization was
sufficient under the tested, declared small-instance conditions.**

The machine-readable gate is [`artifacts/rl_gate_report.json`](artifacts/rl_gate_report.json)
and the rationale is [`docs/rl_gate_decision.md`](docs/rl_gate_decision.md). Gates C
(nonstationary generalization) and E (large-scale scalability) were not tested; that absence is
a limitation, not quantitative permission to add RL.

## Definition-of-done command audit

| Command | Status | Note |
|---|---|---|
| `make setup` | **PASS** | Editable dependency set defined; official SUMO is an explicit optional extra. |
| `make lint` | **PASS** | Ruff clean. |
| `make test` | **PASS** | 38/38 tests. |
| `make benchmark` | **PASS** | Two-route UE/SO and Braess paradox golden checks pass. |
| `make simulation-test` | **PASS** | Actual SUMO ring run and real OSM conversion complete. |
| `make experiment` | **PASS** | Registered analytical smoke run. |
| `make research` | **PASS** | Screening/focused B0–B6 matrix regenerated. |
| `make rl-gate` | **PASS** | Outcome A. |
| `make rl-evaluate` | **PASS** | Explicitly skipped as authorized by Outcome A. |
| `make report` | **PASS** | HTML and figures regenerated from committed evidence. |
| `make audit` | **PASS** | Fast regression, gate, report, and 18 phase reports. |

## Answers to the final research questions

1. **Is preference diversity a coordination resource?** Not established here; H1 failed under
   the declared synthetic generator and route features.
2. **How much Preference Slack is enough?** No general threshold was established. Focused B6
   maximum regret was 0.0444 under a budget of 0.08, yet aggregate H2 still failed.
3. **What navigation penetration prevents ghost jams?** **NOT TESTED.**
4. **Which demand region is most effective?** Analytical figures describe the sampled
   0.8–1.2 range, but no robust general success region supports a claim.
5. **When does adaptive routing fail?** It can lose to ETA-only when preference constraints
   block beneficial routes; B6 enumeration also does not scale beyond its declared limit.
6. **Is there a safety trade-off?** No matched microscopic answer; route-surrogate constraints
   prevented analytical degradation, while H4 remains partial.
7. **Does the mechanism persist on real topology?** **NOT TESTED.** Only topology and conversion
   are verified.
8. **Is RL necessary?** Not under the tested gate conditions.
9. **Where might RL become necessary?** Quantitatively untested nonstationarity or large-scale
   failure could reopen gates C/E; this is a future condition, not a result.
10. **Does RL justify its learning cost?** No evaluation was authorized, so no such claim can
    be made.

The repository therefore passes as an executable, acceptance-safe research system with
reproducible negative and partial findings. It does **not** pass as evidence of real-world
phantom-jam prevention, calibrated human response, crash reduction, or real-network efficacy.
