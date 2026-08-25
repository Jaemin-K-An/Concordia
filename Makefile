PYTHON ?= python3
RUN_CONFIG ?= configs/experiments/smoke.yaml

.PHONY: setup lint test benchmark experiment research rl-gate rl-evaluate conditional-rl report report-v2 phase-reports audit audit-v2 sumo-check sumo-ring-build sumo-ring-run simulation-test phantom-calibrate alignment-study microscopic-study real-topology-study scalability-study drift-study clean

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

clean:
	find src tests -type d -name __pycache__ -prune -exec rm -r {} +
