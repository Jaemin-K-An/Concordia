# CONCORDIA v7 Preregistered Plan

v7 replaces binary SafeMicroSuccess deployment selection with paired conditional treatment-effect
estimation. The primary traffic target is relative `TTT_B1 - TTT_Adaptive`; safety is the paired
adaptive-minus-B1 DRAC-CVaR effect, and utility is maximum affected-user regret.

Development contains 800 immutable historical v6 pairs plus 400 new actual-SUMO pairs. All 1,200
pairs are re-split by seed family into train/calibration/validation roles. No v7 final seed is used
in fitting. After model, intervals, and thresholds are frozen, 300 new paired cases and 12
prespecified OSM OD pairs are evaluated.

The policy intervenes only when traffic uplift lower bound exceeds the frozen threshold, safety
effect upper bound is within margin, regret upper bound is acceptable, and execution is legal.
Validation requires at least 15 selected cases; tiny high-precision subsets are not eligible.
If no non-empty policy achieves precision 0.80 with zero safety violations, v7 freezes safe
abstention and reports Outcome F unchanged. RL remains excluded.
