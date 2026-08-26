# CONCORDIA v8 final research report

CONCORDIA v8 tested a rank-then-veto architecture: the immutable v7 direct
paired random-forest traffic ranker scored candidate benefit, then a calibrated
pre-decision action-aware classifier vetoed the registered unsafe event
`Risk(Adaptive) > Risk(B1) + 0.25`. Predicted regret above 0.08 and illegal
actions remained hard vetoes.

The development corpus contained 2,000 paired conditions, including 500 newly
executed SUMO pairs and 379 unsafe examples. The selected action-aware model
improved validation PR-AUC from 0.476 (state-only) to 0.526 and reduced the
risk-controlled false-safe rate from 0.0164 to 0.0060. Nevertheless, no
non-empty operating point met the registered joint requirements of zero safety
violations, precision at least 0.80, support at least 20, and unsafe recall at
least 0.95. The policy therefore froze safe abstention.

On 400 untouched microscopic pairs, the traffic ranker achieved Spearman 0.407.
Top-10% mean uplift was +0.761%, versus −1.397% for the population. The safety
classifier's final PR-AUC was 0.450. V8-F intervened zero times. Always-on B6
made 400 interventions at 17.75% precision with 91 unsafe outcomes; reconstructed
V7-Mean made 13 interventions at 53.85% precision with one unsafe outcome.

The OSM bridge evaluated 15 new Gangnam OD pairs and 120 paired conditions.
There were 13 safe opportunities, but V8-F recovered none. OSM traffic ranking
reversed (Spearman −0.414), and always-on B6 produced 71 unsafe outcomes. The
OSM study uses real geometry with synthetic demand and preferences.

The registered result is **Outcome F**. The frozen fallback is safe, but v8 does
not support a non-empty adaptive-navigation deployment claim. See
[`FINAL_AUDIT_V8.md`](../FINAL_AUDIT_V8.md) for the complete audit and hypothesis
table.
