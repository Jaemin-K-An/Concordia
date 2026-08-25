#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import tracemalloc
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/concordia-matplotlib-v2")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/concordia-cache-v2")

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from concordia.adaptive import NetworkStateEstimator
from concordia.errors import ConcordiaError
from concordia.evaluation import ExperimentRegistry, capture_source_state
from concordia.models import EdgeData
from concordia.network import RoadNetwork
from concordia.optimization import (
    AdaptiveOptimizer,
    ObjectiveWeights,
    RecedingHorizonOptimizer,
    clustered_greedy_assignment,
)
from concordia.populations import generate_population


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiments" / "scalability.yaml"
OUTPUT = ROOT / "artifacts" / "studies" / "scalability"


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _parallel_network(candidate_count: int):
    network = RoadNetwork(f"parallel_{candidate_count}")
    for index in range(candidate_count):
        middle = f"M{index}"
        network.add_edge(
            "O",
            middle,
            EdgeData(
                4.5 + 0.25 * index,
                500.0 + 180.0 * index,
                variability=max(0.2, 2.5 - 0.25 * index),
                risk=max(0.01, 0.08 - 0.008 * index),
                complexity=max(0.05, 0.6 - 0.05 * index),
            ),
        )
        network.add_edge(
            middle,
            "D",
            EdgeData(
                4.5 + 0.25 * index,
                500.0 + 180.0 * index,
                variability=max(0.2, 2.5 - 0.25 * index),
                risk=max(0.01, 0.08 - 0.008 * index),
                complexity=max(0.05, 0.6 - 0.05 * index),
            ),
        )
    routes = {
        f"r{index}": network.make_route(f"r{index}", ("O", f"M{index}", "D"))
        for index in range(candidate_count)
    }
    return network, routes


def _run_case(config: dict, user_count: int, candidate_count: int, seed: int) -> dict:
    network, routes = _parallel_network(candidate_count)
    users = generate_population(
        user_count,
        "O",
        "D",
        "high",
        float(config["preference_epsilon"]),
        5.0,
        seed,
    )
    candidates = {user.user_id: tuple(routes) for user in users}
    optimizer = AdaptiveOptimizer(
        network,
        routes,
        objective_weights=ObjectiveWeights(0.01, 1.0, 1.0),
        vehicle_flow=float(config["demand_vehicles_per_hour"]) / user_count,
    )
    private = optimizer.private_best(users, candidates)
    combinations = candidate_count**user_count
    oracle = {
        "status": "NOT_EXECUTED_COMBINATION_GUARD",
        "total_travel_time": None,
        "latency_seconds": None,
    }
    if combinations <= int(config["b6_execution_combination_limit"]):
        oracle_started = time.perf_counter()
        oracle_result = optimizer.exact(
            users,
            candidates,
            safety_delta=0.0,
            max_combinations=int(config["b6_declared_enumeration_limit"]),
        )
        oracle = {
            "status": "EXECUTED",
            "total_travel_time": oracle_result.total_travel_time,
            "latency_seconds": time.perf_counter() - oracle_started,
        }

    tracemalloc.start()
    approximation_started = time.perf_counter()
    approximation = clustered_greedy_assignment(optimizer, users, candidates)
    approximation_latency = time.perf_counter() - approximation_started
    _, approximation_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    b6 = {
        "status": "NOT_EXECUTED_COMBINATION_GUARD",
        "latency_seconds": None,
        "peak_memory_bytes": None,
        "total_travel_time": None,
        "combinations_evaluated": 0,
    }
    if combinations <= int(config["b6_execution_combination_limit"]):
        state = NetworkStateEstimator(network).from_flows(
            optimizer._flows(private.assignments), 0.0
        )
        solver = RecedingHorizonOptimizer(
            network,
            routes,
            float(config["demand_vehicles_per_hour"]) / user_count,
            weights=ObjectiveWeights(0.01, 1.0, 1.0),
            max_combinations=int(config["b6_declared_enumeration_limit"]),
        )
        tracemalloc.start()
        b6_started = time.perf_counter()
        try:
            plan = solver.plan(state, users, candidates, private.assignments)
            b6_assignment = optimizer.evaluate(
                plan.first_assignments, plan.dynamic_regrets
            )
            b6 = {
                "status": "EXECUTED",
                "latency_seconds": time.perf_counter() - b6_started,
                "peak_memory_bytes": None,
                "total_travel_time": b6_assignment.total_travel_time,
                "combinations_evaluated": plan.combinations_evaluated,
            }
        except ConcordiaError as exc:
            b6 = {
                "status": "INFEASIBLE",
                "reason": str(exc),
                "latency_seconds": time.perf_counter() - b6_started,
                "peak_memory_bytes": None,
                "total_travel_time": None,
                "combinations_evaluated": 0,
            }
        finally:
            _, b6_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        b6["peak_memory_bytes"] = b6_peak
    return {
        "seed": seed,
        "user_count": user_count,
        "candidate_count": candidate_count,
        "enumerative_combinations": str(combinations),
        "declared_limit_exceeded": combinations
        > int(config["b6_declared_enumeration_limit"]),
        "B6": b6,
        "exact_regret_safety_oracle": oracle,
        "approximation": {
            "method": approximation.metadata["method"],
            "latency_seconds": approximation_latency,
            "peak_memory_bytes": approximation_peak,
            "total_travel_time": approximation.total_travel_time,
            "max_regret": max(approximation.regrets.values()),
            "safety_limit": approximation.metadata["safety_limit"],
            "total_safety_risk": approximation.total_safety_risk,
            "fallback_used": False,
        },
    }


def _summary(rows: list[dict], config: dict) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["user_count"], row["candidate_count"])].append(row)
    scales = []
    for key, values in sorted(grouped.items()):
        approximation_latency = [item["approximation"]["latency_seconds"] for item in values]
        executed = [item for item in values if item["B6"]["status"] == "EXECUTED"]
        b6_latency = [item["B6"]["latency_seconds"] for item in executed]
        quality_gaps = []
        for item in values:
            reference = (
                item["B6"]["total_travel_time"]
                if item["B6"]["status"] == "EXECUTED"
                else item["exact_regret_safety_oracle"]["total_travel_time"]
            )
            if reference is not None:
                quality_gaps.append(
                    (item["approximation"]["total_travel_time"] - reference) / reference
                )
        scales.append(
            {
                "user_count": key[0],
                "candidate_count": key[1],
                "B6_executed_seed_count": len(executed),
                "B6_p95_latency_seconds": (
                    float(np.percentile(b6_latency, 95)) if b6_latency else None
                ),
                "approximation_median_latency_seconds": float(
                    np.median(approximation_latency)
                ),
                "approximation_p95_latency_seconds": float(
                    np.percentile(approximation_latency, 95)
                ),
                "approximation_max_memory_bytes": max(
                    item["approximation"]["peak_memory_bytes"] for item in values
                ),
                "approximation_mean_gap_vs_exact_or_B6": (
                    float(np.mean(quality_gaps)) if quality_gaps else None
                ),
            }
        )
    operational = [item for item in scales if item["user_count"] >= 100]
    approximation_p95 = max(item["approximation_p95_latency_seconds"] for item in operational)
    quality = [
        item["approximation_mean_gap_vs_exact_or_B6"]
        for item in scales
        if item["approximation_mean_gap_vs_exact_or_B6"] is not None
    ]
    b6_failure = next(
        (
            {"user_count": row["user_count"], "candidate_count": row["candidate_count"]}
            for row in sorted(rows, key=lambda item: (item["user_count"], item["candidate_count"]))
            if row["declared_limit_exceeded"]
        ),
        None,
    )
    residual_problem = bool(
        approximation_p95 > float(config["operational_latency_seconds"])
        or (quality and max(quality) > float(config["approximation_quality_gap_relative"]))
    )
    return {
        "complete": True,
        "study": "Study IV — Scalability / RL Gate E",
        "scale_results": scales,
        "B6_first_declared_limit_failure": b6_failure,
        "B6_enumerative_bottleneck_confirmed": b6_failure is not None,
        "approximation_operational_p95_seconds": approximation_p95,
        "approximation_max_small_scale_gap_vs_exact_or_B6": (
            max(quality) if quality else None
        ),
        "Gate_E": {
            "tested": True,
            "b6_scalability_failure": b6_failure is not None,
            "residual_problem_after_mathematical_approximation": residual_problem,
            "triggered_for_RL": residual_problem,
            "latency_threshold_seconds": config["operational_latency_seconds"],
            "quality_threshold_relative": config["approximation_quality_gap_relative"],
        },
        "claim_boundary": config["claim_boundary"],
    }


def _figures(rows: list[dict]) -> list[Path]:
    directory = OUTPUT / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    outputs = []
    for field, name, ylabel in (
        ("latency_seconds", "solve_time_vs_user_count", "Approximation latency (s)"),
        ("peak_memory_bytes", "memory_vs_user_count", "Peak Python memory (bytes)"),
    ):
        fig, axis = plt.subplots(figsize=(7.2, 4.2))
        for candidate_count in sorted({row["candidate_count"] for row in rows}):
            selected = [row for row in rows if row["candidate_count"] == candidate_count]
            users = sorted({row["user_count"] for row in selected})
            values = [
                np.median(
                    [
                        row["approximation"][field]
                        for row in selected
                        if row["user_count"] == count
                    ]
                )
                for count in users
            ]
            axis.plot(users, values, marker="o", label=f"K={candidate_count}")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("User count N")
        axis.set_ylabel(ylabel)
        axis.legend()
        axis.grid(alpha=0.2)
        fig.tight_layout()
        path = directory / f"{name}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        outputs.append(path)
    return outputs


def run() -> Path:
    source_commit, source_dirty = capture_source_state()
    started = datetime.now(timezone.utc)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    rows = [
        _run_case(config, int(user_count), int(candidate_count), int(seed))
        for user_count in config["user_count"]
        for candidate_count in config["candidate_count"]
        for seed in config["seeds"]
    ]
    summary = _summary(rows, config)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    raw_path = OUTPUT / "raw_metrics.json"
    summary_path = OUTPUT / "summary.json"
    statistics_path = OUTPUT / "statistical_tests.json"
    processed_path = OUTPUT / "processed_metrics.json"
    _write_json(raw_path, rows)
    _write_json(summary_path, summary)
    _write_json(statistics_path, summary["Gate_E"])
    _write_json(processed_path, summary)
    figures = _figures(rows)
    ended = datetime.now(timezone.utc)
    outputs = [raw_path, processed_path, summary_path, statistics_path, *figures]
    run_dir = ExperimentRegistry(str(ROOT / "artifacts" / "runs")).create(
        config,
        summary,
        input_paths=(str(CONFIG.relative_to(ROOT)),),
        external_output_paths=tuple(str(path.relative_to(ROOT)) for path in outputs),
        started_at=started,
        ended_at=ended,
        source_commit=source_commit,
        source_dirty=source_dirty,
    )
    shutil.copyfile(run_dir / "manifest.json", OUTPUT / "manifest.json")
    print(summary_path)
    return summary_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-if-valid", action="store_true")
    arguments = parser.parse_args()
    existing = OUTPUT / "summary.json"
    if arguments.reuse_if_valid and existing.is_file() and json.loads(
        existing.read_text(encoding="utf-8")
    ).get("complete"):
        _write_json(OUTPUT / "processed_metrics.json", json.loads(existing.read_text(encoding="utf-8")))
        print(existing)
    else:
        run()
