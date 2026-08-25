PYTHON ?= python3
RUN_CONFIG ?= configs/experiments/smoke.yaml

.PHONY: setup lint test benchmark experiment report audit sumo-check sumo-ring-build sumo-ring-run clean

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

report:
	PYTHONPATH=src $(PYTHON) -m concordia.cli report --runs artifacts/runs --output artifacts/reports/report.html

audit: lint test benchmark experiment report

sumo-check:
	$(PYTHON) scripts/check_sumo.py

sumo-ring-build: sumo-check
	netconvert --node-files scenarios/sumo/ring/ring.nod.xml --edge-files scenarios/sumo/ring/ring.edg.xml --output-file scenarios/sumo/ring/ring.net.xml

sumo-ring-run: sumo-ring-build
	sumo -c scenarios/sumo/ring/ring.sumocfg --seed 42

clean:
	find src tests -type d -name __pycache__ -prune -exec rm -r {} +
