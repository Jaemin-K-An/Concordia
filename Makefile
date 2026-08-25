PYTHON ?= python3
RUN_CONFIG ?= configs/experiments/smoke.yaml

.PHONY: setup lint test benchmark experiment research rl-gate rl-evaluate conditional-rl report report-v2 phase-reports audit audit-v2 sumo-check sumo-ring-build sumo-ring-run simulation-test phantom-calibrate alignment-study microscopic-study real-topology-study scalability-study drift-study v3-audit v3-dataset feasibility-train feasibility-validate freeze-thresholds v3-holdout v3-microscopic v3-real-topology v3-tail-study v3-report v3-final-audit v4-audit v4-dataset v4-train v4-robust-cv v4-calibrate v4-benefit-model v4-safety-model v4-select-threshold v4-freeze v4-holdout v4-microscopic v4-real-topology v4-stress v4-report v4-final-audit v5-audit v5-dataset v5-regime-discovery v5-train v5-shift-model v5-calibrate v5-micro-dataset v5-micro-correction v5-safety-veto v5-validate v5-freeze v5-holdout v5-microscopic v5-real-topology v5-stress v5-report v5-final-audit clean

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

clean:
	find src tests -type d -name __pycache__ -prune -exec rm -r {} +
