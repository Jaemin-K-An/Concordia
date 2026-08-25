# CONCORDIA v6 Preregistered Plan

v6 replaces analytical-to-microscopic correction with direct prediction of SafeMicroSuccess.
The new actual-SUMO development set has 600 paired conditions split by seed family into
360 train, 120 calibration, and 120 validation pairs. A new 200-pair final micro holdout uses
ten disjoint seeds. Features are restricted to state observed no later than the 30-second
decision timestamp. Model, calibration, feature subset, safety architecture, conformal cutoff,
and threshold are selected before five YAML packages and a SHA-256 manifest are frozen.

Primary targets are micro precision 0.80, coverage 0.10, at least 30 interventions,
Opportunity Recovery Rate 0.40, zero safety violations, and false-safe rate at most 0.05.
The outcome is reported unchanged if these targets fail. RL remains excluded.
