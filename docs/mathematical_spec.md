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
