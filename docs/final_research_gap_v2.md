# CONCORDIA final research gap v2

Baseline re-audit HEAD: `acfbaaa92fe621b0fc43c256d42d80ac2945ce46`  
Audit date: 2026-08-25 (Asia/Seoul)

This document records the evidence boundary before Phase 19–36 work. It does not replace
`FINAL_AUDIT.md`, alter H1/H2/H6, or treat implementation APIs as scientific evidence. Raw
command logs are in `artifacts/current_head_audit/`.

## Current HEAD command re-audit

| Command | Result | Boundary |
|---|---|---|
| `make lint` | PASS | Ruff clean. |
| `make test` | PASS | 38/38 tests. |
| `make benchmark` | PASS | Two-route and Braess golden checks. |
| `make simulation-test` | PASS with localhost socket permission | The sandboxed attempt failed because TraCI could not bind a port and passed `remote-port=None`; the same SUMO/TraCI/sumolib 1.27.1 stack passed outside that socket restriction. |
| `make research` | PASS | Existing analytical matrix regenerated. |
| `make rl-gate` | PASS | Existing Outcome A; Gates C/E remained untested at this baseline. |
| `make report` | PASS | Existing report regenerated. |
| `make audit` | PASS | Existing fast audit does not include the v2 studies. |

## Revalidated gaps

| Gap | Baseline status | Evidence |
|---|---|---|
| A — H3 microscopic phantom-jam prevention | **NOT TESTED** | No matched B1/B6 SUMO event-probability matrix. |
| B — H4 microscopic safety non-degradation | **PARTIAL** | Metrics and one smoke distribution exist; matched non-inferiority does not. |
| C — detector physical wave-speed plausibility | **FAIL** | Two baseline candidates report −76.54 m/s (−275.54 km/h), outside plausible traffic shockwave behavior. They cannot support H3. |
| D — acceptance–traffic fixed point | **NOT TESTED** | MPC uses a one-shot target→acceptance→expected-flow calculation. |
| E — real OSM B1/B6 traffic simulation | **NOT TESTED** | Conversion/topology only; no traffic comparison. |
| F — individual rationality/system efficiency frontier | **NOT TESTED** | No `C*(epsilon)`, PoAlign, marginal value, or knee-point evidence. |

## Physical validation configuration decision

Detector positions are metres increasing downstream and time is seconds. The fitted model is
`position_m = intercept + wave_speed_mps × onset_seconds`; output is recorded in m/s and km/h.
The configurable conservative envelope is 3–25 km/h in absolute backward speed around the
approximately −15 km/h empirical wide-moving-jam value reported in primary traffic-flow work.
The envelope is a falsification filter, not a universal road-class calibration. See
`configs/validation/phantom_detector.yaml`.

The v2 work must therefore count only `VALID` events, keep every rejected recommendation on
its existing route, label real-topology demand as synthetic, and authorize RL only after
measured Gate C/E evidence and the pre-RL scalable mathematical baseline.
