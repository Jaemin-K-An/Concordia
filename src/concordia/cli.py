from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from concordia.config import load_config
from concordia.errors import ConcordiaError
from concordia.evaluation import ExperimentRegistry
from concordia.experiment import run_experiment
from concordia.reporting import build_report
from concordia.scenarios import braess, two_route
from concordia.traffic import TrafficAssignment


def _benchmark() -> dict:
    two_network, two_od, two_demand = two_route()
    two_assignment = TrafficAssignment(two_network)
    two_ue = two_assignment.user_equilibrium({two_od: two_demand})
    two_so = two_assignment.system_optimum({two_od: two_demand})

    base_network, base_od, braess_demand = braess(with_connector=False)
    connected_network, connected_od, _ = braess(with_connector=True)
    base_ue = TrafficAssignment(base_network).user_equilibrium({base_od: braess_demand})
    connected_ue = TrafficAssignment(connected_network).user_equilibrium(
        {connected_od: braess_demand}
    )
    result = {
        "two_route": {
            "ue_ttt": two_ue.total_travel_time,
            "so_ttt": two_so.total_travel_time,
            "poa": two_ue.total_travel_time / two_so.total_travel_time,
            "ue_gap": two_ue.relative_gap,
            "so_gap": two_so.relative_gap,
        },
        "braess": {
            "without_connector_ue_ttt": base_ue.total_travel_time,
            "with_connector_ue_ttt": connected_ue.total_travel_time,
            "paradox_observed": connected_ue.total_travel_time > base_ue.total_travel_time,
            "base_gap": base_ue.relative_gap,
            "connected_gap": connected_ue.relative_gap,
        },
    }
    if not all((two_ue.converged, two_so.converged, base_ue.converged, connected_ue.converged)):
        raise ConcordiaError("one or more benchmark assignments did not converge")
    if not result["braess"]["paradox_observed"]:
        raise ConcordiaError("Braess golden scenario did not exhibit its expected paradox")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="concordia")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("benchmark", help="run deterministic golden benchmarks")
    experiment = subparsers.add_parser("experiment", help="run a registered experiment")
    experiment.add_argument("--config", required=True)
    experiment.add_argument("--runs", default="artifacts/runs")
    report = subparsers.add_parser("report", help="rebuild an HTML report from run records")
    report.add_argument("--runs", default="artifacts/runs")
    report.add_argument("--output", default="artifacts/reports/report.html")
    return parser


def main(argv: Sequence[str] = ()) -> int:
    arguments = build_parser().parse_args(list(argv) if argv else None)
    try:
        if arguments.command == "benchmark":
            print(json.dumps(_benchmark(), indent=2, sort_keys=True))
        elif arguments.command == "experiment":
            config = load_config(arguments.config)
            metrics = run_experiment(config)
            run_dir = ExperimentRegistry(arguments.runs).create(config, metrics)
            display_metrics = {key: value for key, value in metrics.items() if key != "decision_log"}
            print(
                json.dumps(
                    {"run_dir": str(run_dir), "decision_count": len(metrics.get("decision_log", [])), "metrics": display_metrics},
                    indent=2,
                    sort_keys=True,
                )
            )
        elif arguments.command == "report":
            path = build_report(arguments.runs, arguments.output)
            print(str(path))
        return 0
    except ConcordiaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
