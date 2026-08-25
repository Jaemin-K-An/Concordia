# CONCORDIA v5 Deployment Checklist

**Deployment is blocked.** Before any field interpretation:

- [ ] Actual-SUMO intervention precision exceeds 0.50 on a new holdout.
- [ ] Actual-SUMO safety violations are zero and false-safe rate is at most 0.05.
- [ ] At least ten microscopic interventions and one safe success are observed.
- [ ] Real-geometry activation is nonzero on at least six legal OD pairs.
- [ ] Analytical coverage reaches 0.15 while retaining 0.80 precision.
- [ ] The microscopic aggregation defect is fixed before a new freeze.
- [ ] External traffic counts, calibrated demand, and independent safety review are added.

Passing analytical or surrogate checks alone never authorizes deployment.
