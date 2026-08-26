PYTHON ?= python3
RUN_CONFIG ?= configs/experiments/smoke.yaml

.PHONY: setup lint test benchmark experiment research rl-gate rl-evaluate conditional-rl report report-v2 phase-reports audit audit-v2 sumo-check sumo-ring-build sumo-ring-run simulation-test phantom-calibrate alignment-study microscopic-study real-topology-study scalability-study drift-study v3-audit v3-dataset feasibility-train feasibility-validate freeze-thresholds v3-holdout v3-microscopic v3-real-topology v3-tail-study v3-report v3-final-audit v4-audit v4-dataset v4-train v4-robust-cv v4-calibrate v4-benefit-model v4-safety-model v4-select-threshold v4-freeze v4-holdout v4-microscopic v4-real-topology v4-stress v4-report v4-final-audit v5-audit v5-dataset v5-regime-discovery v5-train v5-shift-model v5-calibrate v5-micro-dataset v5-micro-correction v5-safety-veto v5-validate v5-freeze v5-holdout v5-microscopic v5-real-topology v5-stress v5-report v5-final-audit v6-audit v6-micro-design v6-micro-dataset v6-label v6-train v6-temporal-model v6-safety-model v6-calibrate v6-select-threshold v6-validate v6-freeze v6-analytical-holdout v6-microscopic-holdout v6-real-topology v6-failure-analysis v6-report v6-final-audit clean
.PHONY: v7-audit v7-paired-dataset v7-effect-labels v7-train-uplift v7-train-safety-effect v7-train-regret v7-quantiles v7-conformal v7-validate v7-placebo v7-ablation v7-freeze v7-microscopic-holdout v7-analytical-check v7-real-topology v7-failure-analysis v7-report v7-final-audit
.PHONY: v8-audit v8-safety-dataset v8-action-features v8-train-safety v8-calibrate-safety v8-traffic-ranking-check v8-integrate-policy v8-validate v8-ablation v8-freeze v8-microscopic-holdout v8-real-topology v8-failure-analysis v8-report v8-final-audit
.PHONY: v9-audit v9-preregister v9-action-space v9-actionability v9-train-surrogate v9-train-safety v9-rollout-validation v9-policy-validation v9-repair v9-freeze v9-micro-holdout v9-mismatch-stress v9-real-topology v9-failure-analysis v9-report v9-final-audit
.PHONY: v10-audit v10-preregister v10-racing-engine v10-development v10-validation v10-repair v10-freeze v10-micro-holdout v10-report v10-final-audit

setup:
	$(PYTHON) -m pip install -e '.[dev,analysis]'

lint:
	$(PYTHON) -m ruff check src tests scripts

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -p 'test_*.py' -v

benchmark:
	PYTHONPATH=src $(PYTHON) -m concordia.cli benchmark

experiment:
	PYTHONPATH=src $(PYTHON) -m concordia.cli experiment --config $(RUN_CONFIG)

research:
	PYTHONPATH=src $(PYTHON) -m concordia.cli research

rl-gate:
	PYTHONPATH=src $(PYTHON) -m concordia.cli rl-gate
	@if [ -f artifacts/studies/scalability/summary.json ] && [ -f artifacts/studies/preference_drift/summary.json ]; then PYTHONPATH=src $(PYTHON) scripts/run_rl_gate_v2.py; else echo "RL Gate v2 pending Phase 31-32 evidence"; fi

rl-evaluate:
	PYTHONPATH=src $(PYTHON) -m concordia.cli rl-evaluate

conditional-rl:
	PYTHONPATH=src $(PYTHON) scripts/run_conditional_rl_study.py

report:
	PYTHONPATH=src $(PYTHON) -m concordia.cli report --runs artifacts/runs --output artifacts/reports/report.html

report-v2:
	PYTHONPATH=src $(PYTHON) scripts/build_final_report_v2.py

phase-reports:
	PYTHONPATH=src $(PYTHON) scripts/build_phase_reports.py

audit: lint test benchmark experiment rl-gate rl-evaluate report phase-reports

audit-v2:
	PYTHONPATH=src $(PYTHON) scripts/run_audit_v2.py
	PYTHONPATH=src $(PYTHON) scripts/build_phase_reports_v2.py

sumo-check:
	$(PYTHON) scripts/check_sumo.py

sumo-ring-build: sumo-check
	PYTHONPATH=src $(PYTHON) scripts/run_sumo_smoke.py --build-only

sumo-ring-run: sumo-ring-build
	PYTHONPATH=src $(PYTHON) scripts/run_sumo_smoke.py

simulation-test:
	PYTHONPATH=src $(PYTHON) scripts/run_sumo_smoke.py
	PYTHONPATH=src $(PYTHON) scripts/verify_real_sumo.py

phantom-calibrate:
	PYTHONPATH=src $(PYTHON) scripts/run_microscopic_study.py --reuse-if-valid

alignment-study:
	PYTHONPATH=src $(PYTHON) scripts/run_alignment_study.py --reuse-if-valid
	PYTHONPATH=src $(PYTHON) scripts/run_fixed_point_ablation.py

microscopic-study:
	PYTHONPATH=src $(PYTHON) scripts/run_microscopic_study.py --reuse-if-valid

real-topology-study:
	PYTHONPATH=src $(PYTHON) scripts/run_real_topology_study.py --reuse-if-valid

scalability-study:
	PYTHONPATH=src $(PYTHON) scripts/run_scalability_study.py --reuse-if-valid

drift-study:
	PYTHONPATH=src $(PYTHON) scripts/run_drift_study.py --reuse-if-valid

v3-audit:
	PYTHONPATH=src $(PYTHON) scripts/run_v3_reaudit.py

v3-dataset:
	PYTHONPATH=src $(PYTHON) scripts/build_v3_dataset.py

feasibility-train:
	PYTHONPATH=src $(PYTHON) scripts/train_v3_feasibility.py

feasibility-validate:
	PYTHONPATH=src $(PYTHON) scripts/validate_v3_feasibility.py

freeze-thresholds:
	PYTHONPATH=src $(PYTHON) scripts/freeze_v3_thresholds.py

v3-holdout:
	PYTHONPATH=src $(PYTHON) scripts/run_v3_holdout.py

v3-microscopic:
	PYTHONPATH=src:scripts $(PYTHON) scripts/run_v3_microscopic.py

v3-real-topology:
	PYTHONPATH=src:scripts $(PYTHON) scripts/run_v3_real_topology.py

v3-tail-study:
	PYTHONPATH=src $(PYTHON) scripts/run_v3_tail_study.py

v3-report:
	PYTHONPATH=src $(PYTHON) scripts/build_final_report_v3.py

v3-final-audit:
	PYTHONPATH=src $(PYTHON) scripts/run_v3_final_audit.py

v4-audit:
	PYTHONPATH=src $(PYTHON) scripts/run_v4_audit.py

v4-dataset:
	PYTHONPATH=src $(PYTHON) scripts/build_v4_dataset.py

v4-train:
	PYTHONPATH=src $(PYTHON) scripts/train_v4_models.py

v4-robust-cv:
	PYTHONPATH=src $(PYTHON) scripts/run_v4_robust_cv.py

v4-calibrate:
	PYTHONPATH=src $(PYTHON) scripts/calibrate_v4_models.py

v4-benefit-model:
	PYTHONPATH=src $(PYTHON) scripts/train_v4_benefit.py

v4-safety-model:
	PYTHONPATH=src $(PYTHON) scripts/train_v4_safety.py

v4-select-threshold:
	PYTHONPATH=src $(PYTHON) scripts/select_v4_threshold.py

v4-freeze:
	PYTHONPATH=src $(PYTHON) scripts/freeze_v4.py

v4-holdout:
	PYTHONPATH=src $(PYTHON) scripts/run_v4_holdout.py

v4-microscopic:
	PYTHONPATH=src:scripts $(PYTHON) scripts/run_v4_microscopic.py

v4-real-topology:
	PYTHONPATH=src:scripts $(PYTHON) scripts/run_v4_real_topology.py

v4-stress:
	PYTHONPATH=src $(PYTHON) scripts/run_v4_stress.py

v4-report:
	PYTHONPATH=src $(PYTHON) scripts/build_final_report_v4.py

v4-final-audit:
	PYTHONPATH=src $(PYTHON) scripts/run_v4_final_audit.py

v5-audit:
	PYTHONPATH=src $(PYTHON) scripts/run_v5_audit.py

v5-dataset:
	PYTHONPATH=src $(PYTHON) scripts/build_v5_dataset.py

v5-regime-discovery:
	PYTHONPATH=src $(PYTHON) scripts/discover_v5_regimes.py

v5-train:
	PYTHONPATH=src $(PYTHON) scripts/train_v5_models.py

v5-shift-model:
	PYTHONPATH=src $(PYTHON) scripts/train_v5_shift.py

v5-calibrate:
	PYTHONPATH=src $(PYTHON) scripts/calibrate_v5_models.py

v5-micro-dataset:
	PYTHONPATH=src:scripts $(PYTHON) scripts/build_v5_micro_dataset.py

v5-micro-correction:
	PYTHONPATH=src $(PYTHON) scripts/train_v5_micro_correction.py

v5-safety-veto:
	PYTHONPATH=src $(PYTHON) scripts/train_v5_safety_veto.py

v5-validate:
	PYTHONPATH=src $(PYTHON) scripts/validate_v5.py

v5-freeze:
	PYTHONPATH=src:scripts $(PYTHON) scripts/freeze_v5.py

v5-holdout:
	PYTHONPATH=src:scripts $(PYTHON) scripts/run_v5_holdout.py

v5-microscopic:
	PYTHONPATH=src:scripts $(PYTHON) scripts/run_v5_microscopic.py

v5-real-topology:
	PYTHONPATH=src:scripts $(PYTHON) scripts/run_v5_real_topology.py

v5-stress:
	PYTHONPATH=src:scripts $(PYTHON) scripts/run_v5_stress.py

v5-report:
	PYTHONPATH=src $(PYTHON) scripts/build_final_report_v5.py

v5-final-audit:
	PYTHONPATH=src:scripts $(PYTHON) scripts/run_v5_final_audit.py

v6-audit:
	PYTHONPATH=src:scripts $(PYTHON) scripts/run_v6_audit.py

v6-micro-design:
	PYTHONPATH=src:. $(PYTHON) -m pytest tests/test_v6_micro.py -q

v6-micro-dataset:
	PYTHONPATH=src:scripts $(PYTHON) scripts/build_v6_micro_dataset.py

v6-label: v6-micro-dataset
	PYTHONPATH=src:. $(PYTHON) -m pytest tests/test_v6_micro.py -q

v6-train:
	PYTHONPATH=src $(PYTHON) scripts/train_v6_models.py

v6-temporal-model: v6-train

v6-safety-model: v6-train

v6-calibrate: v6-train

v6-select-threshold: v6-train

v6-validate: v6-train
	PYTHONPATH=src:. $(PYTHON) -m pytest tests/test_v6_micro.py -q

v6-freeze:
	PYTHONPATH=src:scripts $(PYTHON) scripts/freeze_v6.py

v6-analytical-holdout:
	PYTHONPATH=src:scripts $(PYTHON) scripts/run_v6_analytical_holdout.py

v6-microscopic-holdout:
	PYTHONPATH=src:scripts $(PYTHON) scripts/run_v6_microscopic_holdout.py

v6-real-topology:
	PYTHONPATH=src:scripts $(PYTHON) scripts/run_v6_real_topology.py

v6-failure-analysis:
	PYTHONPATH=src:scripts $(PYTHON) scripts/build_v6_failure_analysis.py

v6-report:
	PYTHONPATH=src:scripts $(PYTHON) scripts/build_final_report_v6.py

v6-final-audit:
	PYTHONPATH=src:scripts $(PYTHON) scripts/run_v6_final_audit.py

v7-audit:
	PYTHONPATH=src:scripts $(PYTHON) scripts/run_v7_audit.py

v7-paired-dataset:
	PYTHONPATH=src:scripts $(PYTHON) scripts/build_v7_paired_dataset.py

v7-effect-labels: v7-paired-dataset
	PYTHONPATH=src:. $(PYTHON) -m pytest tests/test_v7_uplift.py -q

v7-train-uplift:
	PYTHONPATH=src:scripts $(PYTHON) scripts/train_v7_uplift.py

v7-train-safety-effect: v7-train-uplift

v7-train-regret: v7-train-uplift

v7-quantiles: v7-train-uplift

v7-conformal: v7-train-uplift

v7-validate: v7-train-uplift
	PYTHONPATH=src:. $(PYTHON) -m pytest tests/test_v7_uplift.py -q

v7-placebo: v7-train-uplift

v7-ablation: v7-train-uplift

v7-freeze:
	PYTHONPATH=src:scripts $(PYTHON) scripts/freeze_v7.py

v7-microscopic-holdout:
	PYTHONPATH=src:scripts $(PYTHON) scripts/run_v7_microscopic_holdout.py

v7-analytical-check:
	PYTHONPATH=src:scripts $(PYTHON) scripts/run_v7_analytical_holdout.py

v7-real-topology:
	PYTHONPATH=src:scripts $(PYTHON) scripts/run_v7_real_topology.py

v7-failure-analysis:
	PYTHONPATH=src:scripts $(PYTHON) scripts/build_v7_failure_analysis.py

v7-report:
	PYTHONPATH=src:scripts $(PYTHON) scripts/build_final_report_v7.py

v7-final-audit:
	PYTHONPATH=src:scripts $(PYTHON) scripts/run_v7_final_audit.py

v8-audit:
	PYTHONPATH=src:scripts $(PYTHON) scripts/run_v8_audit.py

v8-safety-dataset:
	PYTHONPATH=src:scripts $(PYTHON) scripts/build_v8_safety_dataset.py

v8-action-features: v8-safety-dataset
	PYTHONPATH=src:scripts $(PYTHON) -m unittest discover -s tests -p 'test_v8_safety.py' -v

v8-train-safety:
	PYTHONPATH=src:scripts $(PYTHON) scripts/train_v8_safety_policy.py

v8-calibrate-safety: v8-train-safety

v8-traffic-ranking-check: v8-train-safety

v8-integrate-policy: v8-train-safety

v8-validate: v8-train-safety
	PYTHONPATH=src:scripts $(PYTHON) -m unittest discover -s tests -p 'test_v8_safety.py' -v

v8-ablation: v8-train-safety

v8-freeze:
	PYTHONPATH=src:scripts $(PYTHON) scripts/freeze_v8.py

v8-microscopic-holdout:
	PYTHONPATH=src:scripts $(PYTHON) scripts/run_v8_microscopic_holdout.py

v8-real-topology:
	PYTHONPATH=src:scripts $(PYTHON) scripts/run_v8_real_topology.py

v8-failure-analysis:
	PYTHONPATH=src:scripts $(PYTHON) scripts/build_v8_failure_analysis.py

v8-report:
	PYTHONPATH=src:scripts $(PYTHON) scripts/build_final_report_v8.py

v8-final-audit:
	PYTHONPATH=src:scripts $(PYTHON) scripts/run_v8_final_audit.py

v9-preregister:
	PYTHONPATH=src:scripts $(PYTHON) -c "from pathlib import Path; assert Path('configs/v9/preregistration.yaml').is_file(); assert not Path('artifacts/studies/v9_micro_holdout/summary.json').exists(); print('v9 preregistration present; final absent')"

v9-action-space:
	PYTHONPATH=src:scripts $(PYTHON) -m unittest discover -s tests -p 'test_v9_multi_action.py' -v

v9-actionability:
	PYTHONPATH=src:scripts $(PYTHON) scripts/build_v9_actionability.py --workers 8

v9-train-surrogate:
	PYTHONPATH=src:scripts $(PYTHON) scripts/train_v9_surrogate.py

v9-train-safety:
	PYTHONPATH=src:scripts $(PYTHON) scripts/validate_v9_optimizer.py

v9-rollout-validation:
	PYTHONPATH=src:scripts $(PYTHON) scripts/validate_v9_optimizer.py

v9-policy-validation:
	PYTHONPATH=src:scripts $(PYTHON) scripts/validate_v9_optimizer.py

v9-repair:
	PYTHONPATH=src:scripts $(PYTHON) scripts/validate_v9_optimizer.py

v10-preregister:
	PYTHONPATH=src:scripts $(PYTHON) -c "from pathlib import Path; assert Path('configs/v10/preregistration.yaml').is_file(); assert not Path('artifacts/studies/v10_micro_holdout/summary.json').exists(); assert not Path('artifacts/v10/final_seed_manifest.json').exists(); print('v10 preregistration present; final absent')"

clean:
	find src tests -type d -name __pycache__ -prune -exec rm -r {} +
