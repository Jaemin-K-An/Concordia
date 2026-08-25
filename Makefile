PYTHON ?= python3
RUN_CONFIG ?= configs/experiments/smoke.yaml

.PHONY: setup lint test benchmark experiment research rl-gate rl-evaluate report phase-reports audit sumo-check sumo-ring-build sumo-ring-run simulation-test clean

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

rl-evaluate:
	PYTHONPATH=src $(PYTHON) -m concordia.cli rl-evaluate

report:
	PYTHONPATH=src $(PYTHON) -m concordia.cli report --runs artifacts/runs --output artifacts/reports/report.html

phase-reports:
	PYTHONPATH=src $(PYTHON) scripts/build_phase_reports.py

audit: lint test benchmark experiment rl-gate rl-evaluate report phase-reports

sumo-check:
	$(PYTHON) scripts/check_sumo.py

sumo-ring-build: sumo-check
	PYTHONPATH=src $(PYTHON) scripts/run_sumo_smoke.py --build-only

sumo-ring-run: sumo-ring-build
	PYTHONPATH=src $(PYTHON) scripts/run_sumo_smoke.py

simulation-test:
	PYTHONPATH=src $(PYTHON) scripts/run_sumo_smoke.py
	PYTHONPATH=src $(PYTHON) scripts/verify_real_sumo.py

clean:
	find src tests -type d -name __pycache__ -prune -exec rm -r {} +
