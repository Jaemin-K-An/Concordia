from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import yaml

from concordia.config import load_config
from concordia.errors import ConcordiaError
from concordia.evaluation import ExperimentRegistry, capture_source_state
from concordia.experiment import run_experiment
from concordia.reporting import build_report
from concordia.research import run_research_matrix, write_research_summary
from concordia.rl_gate import evaluate_rl_gate, write_rl_gate
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
    research = subparsers.add_parser("research", help="run screening and focused B0-B6 matrix")
    research.add_argument("--config", default="configs/experiments/research_matrix.yaml")
    research.add_argument("--output", default="artifacts/studies/analytical_matrix/summary.json")
    gate = subparsers.add_parser("rl-gate", help="evaluate the mandatory RL introduction gate")
    gate.add_argument("--research", default="artifacts/studies/analytical_matrix/summary.json")
    gate.add_argument("--output", default="artifacts/rl_gate_report.json")
    gate.add_argument("--document", default="docs/rl_gate_decision.md")
    subparsers.add_parser("rl-evaluate", help="evaluate RL or record the gate-authorized skip")
    return parser


def main(argv: Sequence[str] = ()) -> int:
    arguments = build_parser().parse_args(list(argv) if argv else None)
    try:
        if arguments.command == "benchmark":
            print(json.dumps(_benchmark(), indent=2, sort_keys=True))
        elif arguments.command == "experiment":
            config = load_config(arguments.config)
            source_commit, source_dirty = capture_source_state()
            started = datetime.now(timezone.utc)
            metrics = run_experiment(config)
            ended = datetime.now(timezone.utc)
            run_dir = ExperimentRegistry(arguments.runs).create(
                config,
                metrics,
                input_paths=(arguments.config,),
                started_at=started,
                ended_at=ended,
                source_commit=source_commit,
                source_dirty=source_dirty,
            )
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
        elif arguments.command == "research":
            with Path(arguments.config).open("r", encoding="utf-8") as handle:
                research_config = yaml.safe_load(handle)
            if not isinstance(research_config, dict) or not research_config.get("seeds"):
                raise ConcordiaError("research matrix is malformed")
            source_commit, source_dirty = capture_source_state()
            started = datetime.now(timezone.utc)
            summary = run_research_matrix(research_config)
            ended = datetime.now(timezone.utc)
            path = write_research_summary(summary, arguments.output)
            run_dir = ExperimentRegistry().create(
                research_config,
                {key: value for key, value in summary.items() if key not in {"screening", "focused"}},
                input_paths=(arguments.config,),
                external_output_paths=(arguments.output,),
                started_at=started,
                ended_at=ended,
                source_commit=source_commit,
                source_dirty=source_dirty,
            )
            study_manifest = Path(arguments.output).with_name("manifest.json")
            shutil.copyfile(run_dir / "manifest.json", study_manifest)
            print(json.dumps({"summary": str(path), "run_dir": str(run_dir)}, indent=2))
        elif arguments.command == "rl-gate":
            with Path(arguments.research).open("r", encoding="utf-8") as handle:
                research_summary = json.load(handle)
            result = evaluate_rl_gate(research_summary)
            write_rl_gate(result, arguments.output, arguments.document)
            print(json.dumps(result, indent=2, sort_keys=True))
        elif arguments.command == "rl-evaluate":
            path = Path("artifacts/rl_gate_report.json")
            if not path.is_file():
                raise ConcordiaError("run the mandatory RL gate before RL evaluation")
            result = json.loads(path.read_text(encoding="utf-8"))
            if result.get("rl_authorized") and not result.get("rl_introduced"):
                raise ConcordiaError("RL evaluation implementation is required after a passing gate")
            print("RL evaluation skipped: Outcome A authorized no RL implementation.")
        return 0
    except ConcordiaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
