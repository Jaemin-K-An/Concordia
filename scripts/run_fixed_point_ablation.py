#!/usr/bin/env python3
from __future__ import annotations

import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from concordia.behavior import AcceptanceModel
from concordia.evaluation import ExperimentRegistry, capture_source_state
from concordia.optimization import (
    AcceptanceTrafficFixedPointSolver,
    AdaptiveOptimizer,
)
from concordia.populations import generate_population
from concordia.research import SCENARIOS


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "studies" / "fixed_point_ablation"
SEEDS = [11, 23, 37, 53, 71]


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _ttt(network, flows):
    return sum(
        flow * network.edge_data(edge).travel_time(flow) for edge, flow in flows.items()
    )


def run() -> Path:
    source_commit, source_dirty = capture_source_state()
    started = datetime.now(timezone.utc)
    rows = []
    for scenario in ("two_route", "merge", "signalized"):
        network, od, demand = SCENARIOS[scenario]()
        routes = {
            route.route_id: route
            for route in network.multiobjective_candidate_routes(
                *od, k_per_objective=4, max_overlap=1.0, pareto_filter=False
            )
        }
        for seed in SEEDS:
            users = generate_population(6, *od, "high", 0.08, 5.0, seed)
            candidates = {user.user_id: tuple(routes) for user in users}
            optimizer = AdaptiveOptimizer(network, routes, vehicle_flow=demand / len(users))
            current = optimizer.private_best(users, candidates)
            proposed = optimizer.greedy_vde(users, candidates)
            solver = AcceptanceTrafficFixedPointSolver(
                network,
                routes,
                demand / len(users),
                acceptance_model=AcceptanceModel(),
            )
            initial_flows = optimizer._flows(current.assignments)
            fp0 = solver.solve(
                initial_flows,
                users,
                candidates,
                current.assignments,
                proposed.assignments,
                max_iterations=1,
            )
            fp1 = solver.solve(
                initial_flows,
                users,
                candidates,
                current.assignments,
                proposed.assignments,
            )
            edge_errors = [
                abs(fp0.expected_flows[edge] - fp1.expected_flows[edge])
                for edge in network.edges
            ]
            eta_errors = []
            for route in routes.values():
                fp0_eta = network.path_features(route.nodes, fp0.expected_flows).time
                fp1_eta = network.path_features(route.nodes, fp1.expected_flows).time
                eta_errors.append(abs(fp0_eta - fp1_eta))
            rng = random.Random(seed)
            outcomes = {
                user.user_id: int(
                    rng.random() <= fp1.acceptance_probabilities[user.user_id]
                )
                for user in users
            }
            brier0 = float(
                np.mean(
                    [
                        (fp0.acceptance_probabilities[user.user_id] - outcomes[user.user_id])
                        ** 2
                        for user in users
                    ]
                )
            )
            brier1 = float(
                np.mean(
                    [
                        (fp1.acceptance_probabilities[user.user_id] - outcomes[user.user_id])
                        ** 2
                        for user in users
                    ]
                )
            )
            rows.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "FP0": {
                        "converged": fp0.converged,
                        "iterations": fp0.iterations,
                        "final_residual": fp0.final_residual,
                        "solve_time_seconds": fp0.solve_time_seconds,
                        "acceptance_brier": brier0,
                        "final_ttt": _ttt(network, fp0.expected_flows),
                        "route_reversals": 0,
                    },
                    "FP1": {
                        "converged": fp1.converged,
                        "iterations": fp1.iterations,
                        "final_residual": fp1.final_residual,
                        "solve_time_seconds": fp1.solve_time_seconds,
                        "acceptance_brier": brier1,
                        "final_ttt": _ttt(network, fp1.expected_flows),
                        "route_reversals": 0,
                    },
                    "FP0_vs_FP1": {
                        "mean_eta_prediction_error_minutes": float(np.mean(eta_errors)),
                        "mean_edge_flow_prediction_error_vehicles_per_hour": float(
                            np.mean(edge_errors)
                        ),
                    },
                }
            )
    full_converged = [row for row in rows if row["FP1"]["converged"]]
    summary = {
        "complete": len(full_converged) == len(rows),
        "study": "FP0 one-shot vs FP1 acceptance–traffic fixed point",
        "run_count": len(rows),
        "FP1_converged_count": len(full_converged),
        "FP1_nonconverged_count": len(rows) - len(full_converged),
        "mean_eta_prediction_error_minutes": float(
            np.mean(
                [
                    row["FP0_vs_FP1"]["mean_eta_prediction_error_minutes"]
                    for row in rows
                ]
            )
        ),
        "mean_flow_prediction_error_vehicles_per_hour": float(
            np.mean(
                [
                    row["FP0_vs_FP1"][
                        "mean_edge_flow_prediction_error_vehicles_per_hour"
                    ]
                    for row in rows
                ]
            )
        ),
        "mean_FP0_acceptance_brier": float(
            np.mean([row["FP0"]["acceptance_brier"] for row in rows])
        ),
        "mean_FP1_acceptance_brier": float(
            np.mean([row["FP1"]["acceptance_brier"] for row in rows])
        ),
        "mean_FP0_solve_time_seconds": float(
            np.mean([row["FP0"]["solve_time_seconds"] for row in rows])
        ),
        "mean_FP1_solve_time_seconds": float(
            np.mean([row["FP1"]["solve_time_seconds"] for row in rows])
        ),
        "route_reversals_note": "single-plan ablation; both are zero by construction",
        "claim_boundary": (
            "FP1 is the analytical expected-flow reference, not observed human acceptance."
        ),
    }
    raw_path = OUTPUT / "raw_metrics.json"
    summary_path = OUTPUT / "summary.json"
    _write(raw_path, rows)
    _write(summary_path, summary)
    ended = datetime.now(timezone.utc)
    run_dir = ExperimentRegistry(str(ROOT / "artifacts" / "runs")).create(
        {"study": "fixed_point_ablation", "seeds": SEEDS},
        summary,
        external_output_paths=(
            str(raw_path.relative_to(ROOT)),
            str(summary_path.relative_to(ROOT)),
        ),
        started_at=started,
        ended_at=ended,
        source_commit=source_commit,
        source_dirty=source_dirty,
    )
    shutil.copyfile(run_dir / "manifest.json", OUTPUT / "manifest.json")
    print(summary_path)
    return summary_path


if __name__ == "__main__":
    run()
