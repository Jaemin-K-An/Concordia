# CONCORDIA v5 Methodology

v5 separates development into analytical training, calibration, validation, shift validation,
micro development, micro calibration, and micro validation. Final analytical, stress,
microscopic, and OSM seeds are disjoint. A learned regime router uses navigation penetration and
structural route features; a robust median/MAD shift detector produces DSS and three shift cells.
The analytical primary policy is V5-RD. The actual-SUMO bridge adds a benefit correction,
microscopic success calibration, and bootstrap-logistic safety UCB veto.

Five deployment YAML packages and a SHA-256 manifest were committed before holdout generation.
Strong shift always abstains. The calibration protocol is fixed to ten equal-width bins on [0,1].
All success labels require legal execution, bounded regret/safety, and at least 1% relative TTT gain.
RL is excluded.
