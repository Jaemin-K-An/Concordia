# CONCORDIA v5 Failure Analysis

The analytical primary policy passed 80% precision but missed 15% coverage. The stress target
passed, although DSS did not improve precision over the no-DSS ablation. The micro correction
worsened final benefit MAE from
0.054549 to
0.054696.

In 100 actual-SUMO final pairs, V5-F intervened 10 times, succeeded once, and admitted one
surrogate-safety failure. False-safe rate was 10%. On six stratified OSM OD pairs and 48 paired
conditions it abstained everywhere. The next version should treat microscopic safety and benefit
as first-class causal targets, enlarge seed-disjoint micro development data, and require a
nonzero activation validation guard before freeze.
