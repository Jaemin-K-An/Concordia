#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/concordia-matplotlib-v2")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/concordia-cache-v2")

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from concordia.evaluation import ExperimentRegistry, capture_source_state, paired_comparison
from concordia.models import PreferenceVector, User
from concordia.optimization import (
    AdaptiveOptimizer,
    ObjectiveWeights,
    clustered_greedy_assignment,
)
from concordia.populations import generate_population
from concordia.preferences import UtilityModel, preference_slack
from concordia.research import SCENARIOS


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiments" / "preference_drift.yaml"
OUTPUT = ROOT / "artifacts" / "studies" / "preference_drift"


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _shift(user: User, config: dict) -> User:
    weights = user.preferences.normalized().as_dict()
    shift = config["phase_2_preference_shift"]
    weights["time"] *= float(shift["time_multiplier"])
    weights["variability"] *= float(shift["variability_multiplier"])
    weights["risk"] *= float(shift["risk_multiplier"])
    total = sum(weights.values())
    return User(
        user.user_id,
        user.origin,
        user.destination,
        PreferenceVector(**{key: value / total for key, value in weights.items()}),
        user.epsilon,
        user.rationality,
    )


def _regrets(users, candidates, routes, assignments):
    model = UtilityModel()
    output = {}
    for user in users:
        utilities = model.utilities(
            user.preferences, (routes[route_id] for route_id in candidates[user.user_id])
        )
        output[user.user_id] = preference_slack(utilities)[assignments[user.user_id]]
    return output


def _run_case(config: dict, scenario: str, seed: int) -> dict:
    network, od, base_demand = SCENARIOS[scenario]()
    routes = {
        route.route_id: route
        for route in network.multiobjective_candidate_routes(
            *od, k_per_objective=4, max_overlap=1.0, pareto_filter=False
        )
    }
    phase_1_users = generate_population(
        int(config["user_count"]),
        *od,
        "high",
        float(config["preference_epsilon"]),
        5.0,
        seed,
    )
    phase_2_users = [_shift(user, config) for user in phase_1_users]
    candidates = {user.user_id: tuple(routes) for user in phase_1_users}
    weights = ObjectiveWeights(0.01, 1.0, 1.0)
    phase_1_optimizer = AdaptiveOptimizer(
        network,
        routes,
        objective_weights=weights,
        vehicle_flow=(
            base_demand * float(config["phase_1_demand_scale"]) / len(phase_1_users)
        ),
    )
    phase_2_optimizer = AdaptiveOptimizer(
        network,
        routes,
        objective_weights=weights,
        vehicle_flow=(
            base_demand * float(config["phase_2_demand_scale"]) / len(phase_2_users)
        ),
    )
    phase_1 = clustered_greedy_assignment(
        phase_1_optimizer, phase_1_users, candidates
    )
    phase_1_oracle = phase_1_optimizer.exact(
        phase_1_users,
        candidates,
        safety_delta=1e9,
        max_combinations=100_000,
    )
    frozen_regrets = _regrets(
        phase_2_users, candidates, routes, phase_1.assignments
    )
    frozen = phase_2_optimizer.evaluate(phase_1.assignments, frozen_regrets)
    adaptive = clustered_greedy_assignment(
        phase_2_optimizer, phase_2_users, candidates
    )
    oracle = phase_2_optimizer.exact(
        phase_2_users,
        candidates,
        safety_delta=1e9,
        max_combinations=100_000,
    )
    frozen_violation = float(
        np.mean(
            [
                max(0.0, frozen_regrets[user.user_id] - user.epsilon)
                for user in phase_2_users
            ]
        )
    )
    adaptive_violation = float(
        np.mean(
            [
                max(0.0, adaptive.regrets[user.user_id] - user.epsilon)
                for user in phase_2_users
            ]
        )
    )
    phase_1_gap = (
        phase_1.total_travel_time - phase_1_oracle.total_travel_time
    ) / phase_1_oracle.total_travel_time
    phase_2_gap = (
        adaptive.total_travel_time - oracle.total_travel_time
    ) / oracle.total_travel_time
    return {
        "scenario": scenario,
        "seed": seed,
        "phase_1_ttt": phase_1.total_travel_time,
        "phase_1_oracle_ttt": phase_1_oracle.total_travel_time,
        "phase_1_relative_degradation": phase_1_gap,
        "phase_2_oracle_ttt": oracle.total_travel_time,
        "frozen_phase_2_ttt": frozen.total_travel_time,
        "adaptive_phase_2_ttt": adaptive.total_travel_time,
        "frozen_relative_degradation": (
            frozen.total_travel_time - oracle.total_travel_time
        )
        / oracle.total_travel_time,
        "adaptive_relative_degradation": phase_2_gap,
        "nonstationarity_incremental_degradation": max(0.0, phase_2_gap - phase_1_gap),
        "frozen_mean_regret_violation": frozen_violation,
        "adaptive_mean_regret_violation": adaptive_violation,
        "frozen_max_regret": max(frozen_regrets.values()),
        "adaptive_max_regret": max(adaptive.regrets.values()),
        "route_replanned": phase_1.assignments != adaptive.assignments,
    }


def _summary(rows: list[dict], config: dict) -> dict:
    frozen = [row["frozen_relative_degradation"] for row in rows]
    adaptive = [row["adaptive_relative_degradation"] for row in rows]
    incremental = [row["nonstationarity_incremental_degradation"] for row in rows]
    threshold = float(config["gate_c_degradation_threshold_relative"])
    aggregation = str(config.get("gate_c_aggregation", "median"))
    if aggregation != "median":
        raise ValueError("Gate C currently requires the pre-registered median aggregation")
    residual = float(np.median(incremental))
    return {
        "complete": True,
        "study": "Study IV — Preference Drift / RL Gate C",
        "traffic_level_case_count": len(rows),
        "frozen_median_relative_degradation": float(np.median(frozen)),
        "median_nonstationarity_incremental_degradation": residual,
        "gate_c_aggregation": aggregation,
        "adaptive_absolute_median_gap_vs_oracle": float(np.median(adaptive)),
        "adaptive_p95_incremental_degradation": float(np.percentile(incremental, 95)),
        "paired_adaptation_effect": paired_comparison(frozen, adaptive, seed=61),
        "frozen_regret_violation_case_count": sum(
            row["frozen_mean_regret_violation"] > 0 for row in rows
        ),
        "adaptive_regret_violation_case_count": sum(
            row["adaptive_mean_regret_violation"] > 1e-10 for row in rows
        ),
        "Gate_C": {
            "tested": True,
            "threshold_relative": threshold,
            "measured_median_nonstationarity_incremental_degradation": residual,
            "triggered_for_RL": residual > threshold,
        },
        "claim_boundary": config["claim_boundary"],
    }


def _figure(rows: list[dict]) -> Path:
    directory = OUTPUT / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    labels = [f"{row['scenario']}:{row['seed']}" for row in rows]
    x = np.arange(len(rows))
    axis.plot(
        x,
        [row["frozen_relative_degradation"] for row in rows],
        marker="o",
        label="frozen",
        color="#999999",
    )
    axis.plot(
        x,
        [row["adaptive_relative_degradation"] for row in rows],
        marker="o",
        label="replanned",
        color="#111111",
    )
    axis.plot(
        x,
        [row["phase_1_relative_degradation"] for row in rows],
        marker=".",
        label="phase-1 approximation gap",
        color="#444444",
        linestyle=":",
    )
    axis.axhline(0.10, color="#555555", linestyle="--", label="Gate C 10%")
    axis.set_xticks(x, labels, rotation=75, fontsize=7)
    axis.set_ylabel("Relative degradation vs phase-2 oracle")
    axis.legend()
    axis.grid(alpha=0.2)
    fig.tight_layout()
    path = directory / "preference_drift_performance.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def run() -> Path:
    source_commit, source_dirty = capture_source_state()
    started = datetime.now(timezone.utc)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    rows = [
        _run_case(config, str(scenario), int(seed))
        for scenario in config["scenarios"]
        for seed in config["seeds"]
    ]
    summary = _summary(rows, config)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    raw_path = OUTPUT / "raw_metrics.json"
    summary_path = OUTPUT / "summary.json"
    statistics_path = OUTPUT / "statistical_tests.json"
    figure = _figure(rows)
    _write_json(raw_path, rows)
    _write_json(summary_path, summary)
    _write_json(statistics_path, summary["paired_adaptation_effect"])
    ended = datetime.now(timezone.utc)
    outputs = [raw_path, summary_path, statistics_path, figure]
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
        print(existing)
    else:
        run()
