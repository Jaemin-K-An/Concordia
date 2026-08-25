# Mathematical specification

## 1. Scope and intervention boundary

Concordia recommends a legal, feasible route. It does not issue longitudinal or lateral
vehicle-control commands. Route attributes shown to a user are computed features; no false
ETA, incident, cost, or safety claim is permitted. A recommendation is admissible only when
its estimated private-utility regret and network safety constraints pass as hard checks.

## 2. Sets, variables, and units

Let the directed physical graph be \(G=(V,E)\), OD pairs be \(k\in K\), candidate routes be
\(r\in R_k\), and users be \(i\in I\). Edge flow \(x_e\) is vehicles/hour, capacity \(c_e\)
is vehicles/hour, free-flow time \(t^0_e\) is minutes, length is kilometres, cost is KRW,
and risk/complexity/familiarity are dimensionless indices in \([0,1]\).

The BPR link model used by the deterministic harness is

\[
t_e(x_e)=t^0_e\left(1+\alpha_e(x_e/c_e)^{\beta_e}\right).
\]

It is a static analytical baseline, not a model of stop-and-go dynamics. SUMO supplies the
time-varying microscopic layer in later experiments.

## 3. Route representation and normalization

Each route has a raw feature vector

\[
z_r=[T_r,\sigma(T_r),C_r,R_r,M_r,F_r].
\]

Utility uses configured positive scale constants \(s_j\):

\[
U_i(r)=-w_{T,i}T_r/s_T-w_{V,i}\sigma(T_r)/s_V-w_{C,i}C_r/s_C
-w_{R,i}R_r/s_R-w_{M,i}M_r/s_M+w_{F,i}F_r/s_F.
\]

Weights are non-negative and normalized to sum to one. Default scales are experiment inputs,
not learned facts. Conclusions must include scale and population sensitivity analyses.

Candidate paths are generated in increasing generalized edge cost. Paths exceeding the
configured maximum edge-overlap coefficient are removed, followed optionally by removal of
Pareto-dominated paths over time, variability, cost, risk, and complexity (familiarity is
maximized).

## 4. Choice, slack, and truthful individual rationality

The deterministic model selects \(r_i^*=\arg\max_r U_i(r)\). The bounded-rational model is

\[
P_i(r)=\exp(\lambda_i U_i(r))/\sum_q\exp(\lambda_i U_i(q)),
\]

implemented with log-sum-exp stabilization. Preference Slack and realized regret are

\[
PS_{i,r}=U_i(r_i^*)-U_i(r),\qquad Regret_i=PS_{i,r_i^{rec}}.
\]

An admissible route satisfies \(Regret_i\le\epsilon_i+\tau\). In no-sacrifice mode,
\(\epsilon_i=0\); \(\tau\) is a numerical tolerance, not an extra utility allowance.

## 5. Equilibrium and social optimum baselines

Wardrop UE minimizes the Beckmann potential for separable monotone costs:

\[
\min_x\sum_e\int_0^{x_e}t_e(w)dw.
\]

SO minimizes total system travel time:

\[
\min_x C(x)=\sum_e x_et_e(x_e).
\]

Both are solved by deterministic Frank-Wolfe all-or-nothing assignment. UE uses current link
cost and SO uses marginal social cost \(t_e+x_e t'_e\). Convergence is reported using the
relative gap; non-convergence is never silently accepted. Price of Anarchy is
\(PoA=C(x^{UE})/C(x^{SO})\).

## 6. Aligned assignment

For assigning user \(i\) to candidate \(r\), marginal network benefit is measured against the
user's private-best reference assignment:

\[
MNB_{i,r}=C(x\mid i\rightarrow r_i^*)-C(x\mid i\rightarrow r).
\]

The exploration score is \(VDE_{i,r}=MNB_{i,r}/(PS_{i,r}+\varepsilon_0)\). It is only a greedy
baseline. For small populations the exact oracle enumerates all feasible route combinations
and minimizes

\[
J=TTT+\lambda_G G+\lambda_R R+\lambda_C HHI,
\]

where \(G\) is aggregate ghost-jam risk, \(R\) is aggregate route safety exposure, and HHI is
the route-share concentration \(\sum_r p_r^2\). Entropy is reported separately. The exact
oracle enforces individual regret, path feasibility, and

\[
R_{adaptive}\le R_{baseline}+\delta.
\]

For larger populations the greedy policy is explicitly labelled approximate.

## 7. Ghost-jam and safety measurements

The analytical ghost-risk score is a calibrated-input placeholder combining saturation,
speed coefficient of variation, and acceleration variance through a logistic link. It is not
called an observed phantom jam. A microscopic phantom-jam event additionally requires a
sustained oscillation, queue formation, and backward propagation estimate.

For a follower with gap \(g>0\), follower speed \(v_f\), and leader speed \(v_l\):

\[
TTC=g/(v_f-v_l)\quad\text{when }v_f>v_l,
\]

\[
DRAC=(v_f-v_l)^2/(2g).
\]

PET is derived from recorded conflict-zone exit/entry times. The harness reports distributions,
threshold counts, and upper-tail CVaR. These are surrogate measures and are not interpreted as
crash rates.

## 8. Hypotheses and estimands

- H1: holding mean preference fixed, higher variance increases the fraction
  \(P(PS<\epsilon)\) and feasible beneficial diversions.
- H2: aligned assignment reduces TTT subject to bounded regret, including \(\epsilon=0\).
- H3: pre-critical diversion reduces microscopic wave-event probability versus dynamic ETA.
- H4: adaptive safety outcomes are non-inferior within configured \(\delta\).
- H5: variance/tails explain outcomes beyond mean weights.

Primary contrasts compare static shortest path, dynamic ETA, preference-only, SO, greedy VDE,
and exact aligned assignment over matched seeds. Runs report mean, standard deviation, bootstrap
confidence intervals, paired effect sizes, and tail outcomes. A single run is never a research
conclusion.

## 9. Assumptions and limitations

The deterministic graph assumes separable monotone BPR costs, fixed OD demand, and route-level
features supplied truthfully. It cannot establish shockwave, PET, or lane-change findings.
Those claims require trajectory-complete microscopic runs, an explicit event detector, and
calibration. Synthetic preference populations test mechanisms; they do not establish human
behaviour. Real-network results require manifests for source, date, licence, checksum, CRS,
units, and demand provenance.

## 10. Reproducibility contract

Every run records canonical config, UTC timestamp, run ID, Python/platform/dependency versions,
Python/NumPy/SUMO seeds, simulator version, and Git commit. Input validation precedes execution.
Missing SUMO, disconnected OD pairs, empty candidates, invalid probability vectors, negative
flows/features, or infeasible safety constraints raise explicit errors.

## 11. Microscopic network-state semantics

For SUMO edge \(e\), the adapter records vehicle count \(N_e\), mean speed \(v_e\) in m/s,
lane count \(L_e\), lane length \(d_e\) in metres, and occupancy \(o_e\) in percent when
available. The instantaneous density estimate is

\[
k_e=\frac{1000N_e}{L_ed_e}\quad[\text{veh/km/lane}],
\]

and, without an interval detector, the instantaneous traffic-state flow estimate is

\[
q_e=3.6k_ev_e\quad[\text{veh/hour/lane}].
\]

This \(q=kv\) estimate is not reported as counted detector throughput.

## 12. Acceptance-aware receding horizon

At state \(x_t\), each candidate assignment produces a relaxed BPR flow trajectory
\(\hat x_{t+1:t+H}\). Route features and Preference Slack are recomputed on that trajectory.
The objective discounts TTT, analytical ghost-risk exposure, route-risk exposure, and HHI.
Hard filters enforce individual dynamic regret, minimum predicted acceptance, and upper-tail
route-risk CVaR relative to the current baseline plus \(\delta\). Expected flow is a mixture
of proposed and current routes under the synthetic acceptance probability. Only the first
assignment is offered; acceptance is sampled separately and only accepted routes execute.

The current implementation enumerates constant target assignments and is intentionally a
small-instance correctness baseline. It is not a scalable nonlinear stochastic optimizer.
The B5 HiGHS MIP instead minimizes route costs linearized at the observed state, and is
reported separately from the nonlinear exact oracle.

## 13. Predictor, detector, and calibration boundary

`PhantomJamRiskPredictor` features are density, mean speed, speed CV, acceleration variance,
headway variance, flow, saturation, and geometry complexity. Logistic and interpretable
decision-stump models report ROC-AUC, PR-AUC, Brier score, ECE, and false-negative rate.
The event detector independently requires sustained high-density/low-speed episodes at two
or more detectors, sufficient amplitude, and negative fitted propagation speed. The current
SUMO artifact is one synthetic ring smoke run; it exercises the detector but does not provide
the multi-demand train/test calibration or matched prevention probability required for H3.

## 14. Research outcome boundary

The focused B0–B6 matrix uses six synthetic users, matched seeds, and analytical BPR state.
It found no aggregate support for H1 or H2 and no pre-registered stability support for H6.
H5 is exploratory in-sample association only. A signalized failure case shows that ETA-only
can reduce TTT while violating the configured regret budget for some users; this is why
feasibility and network efficiency are reported separately. H3 is NOT TESTED and microscopic
H4 is PARTIAL. These negative/partial findings supersede any generic expectation stated in
Section 8.

## 15. Physical phantom-event validation

Detector position is measured in metres increasing downstream and onset time in seconds. An
EWMA followed by a sustained threshold estimates each detector onset. For a candidate cluster,

\[
x_j=a+v_w t^{onset}_j+\epsilon_j
\]

is fitted directly, so the slope is (v_w) in m/s; km/h is (3.6v_w). Validation records
detector count, (R^2), onset uncertainty, duration, affected length, oscillation amplitude,
minimum speed, density elevation, and queue evidence. The event is classified `VALID`,
`LOW_CONFIDENCE`, `PHYSICALLY_IMPLAUSIBLE`, or `INSUFFICIENT_DETECTORS`. H3 counts only
`VALID`. Speed limits are configuration, not code constants.

## 16. Acceptance–traffic fixed point

For a proposed route assignment, acceptance (Q(\hat x)) depends on the traffic state used to
predict its features, while the expected traffic state is (F(Q)). The final prediction solves

\[
\hat x^*=F(Q(\hat x^*)).
\]

Relaxed Picard iteration uses

\[
\hat x^{k+1}=(1-\eta)\hat x^k+\eta F(Q(\hat x^k))
\]

and stops when the edge-flow (L_\infty) residual is below the configured tolerance. Every
plan records convergence, iterations, residual, and solve time. A non-converged candidate is
not executed. FP0/FP1 evidence separately reports whether the added iteration improves ETA,
acceptance Brier score, expected flow, or TTT enough to justify its compute cost.

## 17. Price of Alignment

Let (C^*(\epsilon)) be the minimum assignment TTT whose individual regret does not exceed
(\epsilon), and let (C_{SO}) be the minimum discrete assignment TTT over the same users and
routes without the alignment constraint. The reported measure is

\[
PoAlign(\epsilon)=\frac{C^*(\epsilon)}{C_{SO}}.
\]

Because the feasible set at (epsilon_1) is contained in that at (epsilon_2\ge\epsilon_1),
(C^*(\epsilon)) must be non-increasing; a controlled regression test enforces this property.
The marginal value is the negative finite difference (-dC^*/d\epsilon). A normalized
curvature calculation selects a knee point without visual judgement. WIN, TRADEOFF, and
INFEASIBLE regions are reported across demand, epsilon, heterogeneity, and penetration rather
than collapsed into a single success rate.

## 18. RL Gate v2 boundary

B6 enumeration failure is measured separately from the residual problem after a hard-
constrained mathematical approximation. Gate E authorizes RL only when that approximation
also misses the frozen five-second latency or five-percent small-scale quality threshold. Gate
C measures the median incremental degradation induced by nonstationarity after subtracting the
same approximation's stationary oracle gap. A high p95 remains a documented failure condition
even when the pre-registered median gate does not trigger.

The final rerun triggered the separately frozen small-scale Gate B after fixed-point iteration
increased exact-cycle runtime. Conditional RL0 therefore used a masked clipped-PPO actor whose
actions were eligibility-epsilon choices only. Invalid regret/safety actions were unavailable.
On held-out analytical conditions it matched, but did not improve upon, the constrained
deterministic minimum-TTT comparator, so it was rejected (Outcome B).
