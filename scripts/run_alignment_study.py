#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/concordia-matplotlib-v2")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/concordia-cache-v2")

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from concordia.alignment import compute_alignment_frontier
from concordia.evaluation import ExperimentRegistry, capture_source_state, summarize_samples
from concordia.populations import generate_population
from concordia.research import SCENARIOS


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiments" / "alignment_frontier.yaml"
OUTPUT = ROOT / "artifacts" / "studies" / "alignment_frontier"


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _figures(rows: list[dict], processed: list[dict], knee_values: list[float]) -> list[Path]:
    figure_dir = OUTPUT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    grouped = defaultdict(list)
    for point in processed:
        grouped[point["epsilon"]].append(point)
    epsilon = np.asarray(sorted(grouped), dtype=float)
    poalign = np.asarray(
        [np.mean([item["price_of_alignment_mean"] for item in grouped[value]]) for value in epsilon]
    )
    ttt = np.asarray(
        [np.mean([item["minimum_feasible_ttt_mean"] for item in grouped[value]]) for value in epsilon]
    )
    marginal = np.asarray(
        [
            np.mean([item["marginal_cost_reduction_mean"] for item in grouped[value]])
            for value in epsilon
        ]
    )

    for name, ylabel, values in (
        ("price_of_alignment", "Price of Alignment", poalign),
        ("efficiency_voluntariness_frontier", "Minimum feasible TTT", ttt),
        ("marginal_value_epsilon", "Marginal TTT reduction / epsilon", marginal),
    ):
        fig, axis = plt.subplots(figsize=(7.2, 4.2))
        axis.plot(epsilon, values, marker="o", color="#111111")
        axis.set_xlabel("Regret budget epsilon")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
        fig.tight_layout()
        path = figure_dir / f"{name}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.hist(knee_values, bins=sorted(set(knee_values + [0.25])), color="#404040")
    axis.set_xlabel("Seed-level knee epsilon")
    axis.set_ylabel("Frontier count")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    path = figure_dir / "knee_point.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    regions = Counter(
        point["alignment_region"] for row in rows for point in row["frontier"]
    )
    fig, axis = plt.subplots(figsize=(6.4, 4.2))
    labels = ["WIN", "TRADEOFF", "INFEASIBLE"]
    axis.bar(labels, [regions[label] for label in labels], color=["#111111", "#666666", "#bbbbbb"])
    axis.set_ylabel("Sampled conditions")
    axis.set_title("Alignment feasibility map")
    fig.tight_layout()
    path = figure_dir / "alignment_feasibility_map.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)
    return paths


def run() -> Path:
    source_commit, source_dirty = capture_source_state()
    started = datetime.now(timezone.utc)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    rows = []
    for scenario in config["scenarios"]:
        network, od, base_demand = SCENARIOS[scenario]()
        routes = {
            route.route_id: route
            for route in network.multiobjective_candidate_routes(
                *od, k_per_objective=4, max_overlap=1.0, pareto_filter=False
            )
        }
        for demand_scale in config["demand_scale"]:
            for heterogeneity in config["heterogeneity"]:
                for seed in config["seeds"]:
                    users = generate_population(
                        int(config["user_count"]),
                        *od,
                        str(heterogeneity),
                        0.0,
                        5.0,
                        int(seed),
                    )
                    candidates = {user.user_id: tuple(routes) for user in users}
                    result = compute_alignment_frontier(
                        network,
                        routes,
                        users,
                        candidates,
                        base_demand * float(demand_scale) / len(users),
                        config["utility_epsilon"],
                        int(config["maximum_combinations"]),
                    )
                    rows.append(
                        {
                            "scenario": scenario,
                            "demand_scale": float(demand_scale),
                            "heterogeneity": heterogeneity,
                            "seed": int(seed),
                            "frontier": [asdict(point) for point in result.points],
                            "unconstrained_system_optimum_ttt": (
                                result.unconstrained_system_optimum_ttt
                            ),
                            "private_best_ttt": result.private_best_ttt,
                            "eta_only_ttt": result.eta_only_ttt,
                            "knee_epsilon": result.knee_epsilon,
                            "knee_index": result.knee_index,
                            "monotonic": result.monotonic,
                            "h1_robustness": dict(result.h1_robustness),
                        }
                    )

    aggregate = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for point in row["frontier"]:
            key = (
                row["scenario"],
                row["demand_scale"],
                row["heterogeneity"],
                point["epsilon"],
            )
            aggregate[key]["price_of_alignment"].append(point["price_of_alignment"])
            aggregate[key]["minimum_feasible_ttt"].append(point["minimum_feasible_ttt"])
            aggregate[key]["marginal"].append(
                point["marginal_cost_reduction_per_epsilon"]
            )
    processed = []
    for key, values in sorted(aggregate.items()):
        poalign = summarize_samples(values["price_of_alignment"], seed=19)
        ttt = summarize_samples(values["minimum_feasible_ttt"], seed=23)
        marginal = summarize_samples(values["marginal"], seed=29)
        processed.append(
            {
                "scenario": key[0],
                "demand_scale": key[1],
                "heterogeneity": key[2],
                "epsilon": key[3],
                "price_of_alignment_mean": poalign["mean"],
                "price_of_alignment_ci95": [poalign["ci95_low"], poalign["ci95_high"]],
                "minimum_feasible_ttt_mean": ttt["mean"],
                "minimum_feasible_ttt_ci95": [ttt["ci95_low"], ttt["ci95_high"]],
                "marginal_cost_reduction_mean": marginal["mean"],
            }
        )

    h1_by_heterogeneity = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for metric, value in row["h1_robustness"].items():
            h1_by_heterogeneity[row["heterogeneity"]][metric].append(value)
    h1_summary = {
        heterogeneity: {
            metric: summarize_samples(values, seed=31)["mean"]
            for metric, values in metrics.items()
        }
        for heterogeneity, metrics in h1_by_heterogeneity.items()
    }
    knee_values = [row["knee_epsilon"] for row in rows]
    region_counts = Counter(
        point["alignment_region"] for row in rows for point in row["frontier"]
    )
    statistical_tests = {
        "frontier_count": len(rows),
        "monotonic_frontier_count": sum(row["monotonic"] for row in rows),
        "monotonicity_violations": [
            {
                key: row[key]
                for key in ("scenario", "demand_scale", "heterogeneity", "seed")
            }
            for row in rows
            if not row["monotonic"]
        ],
        "knee_epsilon": summarize_samples(knee_values, seed=37),
        "phase_counts": dict(region_counts),
        "h1_original_status": "FAIL_UNCHANGED",
        "h1_r_interpretation": (
            "Separate operationalization study; it does not replace the pre-registered H1."
        ),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    raw_path = OUTPUT / "raw_metrics.json"
    processed_path = OUTPUT / "processed_metrics.json"
    statistics_path = OUTPUT / "statistical_tests.json"
    h1_path = OUTPUT / "h1_robustness.json"
    _write_json(raw_path, rows)
    _write_json(processed_path, processed)
    _write_json(statistics_path, statistical_tests)
    _write_json(h1_path, h1_summary)
    figures = _figures(rows, processed, knee_values)
    summary = {
        "complete": True,
        "synthetic": True,
        "study": "Study I — Alignment Frontier",
        "frontier_count": len(rows),
        "sampled_point_count": sum(len(row["frontier"]) for row in rows),
        "monotonicity_violations": len(statistical_tests["monotonicity_violations"]),
        "knee_epsilon": statistical_tests["knee_epsilon"],
        "phase_counts": dict(region_counts),
        "h1_status": "FAIL_UNCHANGED",
        "h2_status": "FAIL_UNCHANGED",
        "claim_boundary": config["claim_boundary"],
    }
    summary_path = OUTPUT / "summary.json"
    _write_json(summary_path, summary)
    ended = datetime.now(timezone.utc)
    outputs = [raw_path, processed_path, statistics_path, h1_path, summary_path, *figures]
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
    checksums = {
        str(path.relative_to(OUTPUT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in outputs
    }
    _write_json(OUTPUT / "output_hashes.json", checksums)
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
