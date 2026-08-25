# CONCORDIA v4 preregistered plan

- Starting HEAD: `b02b9ce62afb59f368547de8e0107f2161797421`
- v2/v3 outcomes remain unchanged; v3 holdout is historical development only.
- Primary constraint: holdout precision ≥0.80 while maximizing coverage.
- Coverage minimum/stretched targets: 0.20/0.25; guard during selection: 0.15.
- Desired/minimum intervention counts: 50/30.
- Strong/very strong lower 95% precision bounds: >0.60/>0.70.
- Per-intervention success: relative TTT gain ≥0.01, regret ≤0.08, safety Δ≤0.25, legal/executable/accepted-only.
- Freeze order: development → robust CV → calibration → validation → commit freeze → new holdout.
- Abstentions are excluded from intervention precision and retained in coverage/PBR denominators.
- RL remains rejected and absent.
