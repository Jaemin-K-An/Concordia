#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/concordia-v6-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/concordia-v6-cache")

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from v6_frozen import verify_frozen, write_json
from v6_micro_sim import build_v6_network, run_v6_pair


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/studies/v6_frozen_micro_holdout"
OUTPUT = ROOT / "artifacts/studies/v6_failure_analysis"


def _failure_type(row: dict) -> str:
    label = row["label"]
    features = row["features_pre_decision"]
    if not label["safety_pass"]:
        return "micro_safety_misclassification"
    if row["counterfactual_adaptive"]["acceptance_rate"] + 0.15 < features["predicted_acceptance"]:
        return "acceptance_mismatch"
    if features["route_overlap"] > 0.65:
        return "route_overlap_failure"
    if features["volume_capacity_ratio"] > 1.15:
        return "secondary_bottleneck"
    if features["navigation_penetration"] < 0.75:
        return "penetration_mismatch"
    if features["flow_instability"] > 0.30:
        return "temporal_instability"
    if features["topology_real_like"] > 0.5:
        return "unseen_topology_shift"
    return "micro_benefit_misclassification"


def _plot(case_id: str, baseline: dict, adaptive: dict) -> list[str]:
    directory = OUTPUT / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    outputs = []
    figure, axes = plt.subplots(5, 1, figsize=(8.0, 10.0), sharex=True)
    fields = (
        ("flow", "Flow (veh/h/lane)"),
        ("speed", "Speed (m/s)"),
        ("queue", "Queue (vehicles)"),
        ("drac_proxy_p95", "DRAC proxy p95"),
        ("minimum_ttc", "Minimum TTC/headway proxy (s)"),
    )
    for run, label, color in ((baseline, "B1", "#4c78a8"), (adaptive, "B6", "#e45756")):
        series = run["diagnostic_series"]
        for axis, (field, title) in zip(axes, fields):
            axis.plot([row["time"] for row in series], [row[field] for row in series], label=label, color=color)
            axis.set_ylabel(title)
            axis.axvline(30.0, color="#333333", linestyle="--", linewidth=0.8)
    axes[0].legend()
    axes[-1].set_xlabel("Simulation time (s)")
    figure.suptitle(f"{case_id}: full paired microscopic diagnostics")
    figure.tight_layout()
    timeseries = directory / f"{case_id}_timeseries.png"
    figure.savefig(timeseries, dpi=170)
    plt.close(figure)
    outputs.append(str(timeseries.relative_to(ROOT)))

    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    names = ("diverted_vehicle_count", "offer_count", "accepted_count", "rejected_count")
    x = np.arange(len(names))
    axis.bar(x - 0.18, [baseline[name] for name in names], 0.36, label="B1")
    axis.bar(x + 0.18, [adaptive[name] for name in names], 0.36, label="B6")
    axis.set_xticks(x, [name.replace("_count", "") for name in names], rotation=15)
    axis.set_ylabel("Vehicles")
    axis.set_title("Route recommendation and realized load signals")
    axis.legend()
    figure.tight_layout()
    route_load = directory / f"{case_id}_route_load.png"
    figure.savefig(route_load, dpi=170)
    plt.close(figure)
    outputs.append(str(route_load.relative_to(ROOT)))

    rows = adaptive["spatiotemporal"]
    times = sorted({row["time"] for row in rows})
    positions = sorted({row["position_index"] for row in rows})
    speed = np.full((len(positions), len(times)), np.nan)
    density = np.full_like(speed, np.nan)
    time_index = {value: index for index, value in enumerate(times)}
    position_index = {value: index for index, value in enumerate(positions)}
    for row in rows:
        i = position_index[row["position_index"]]
        j = time_index[row["time"]]
        speed[i, j] = row["speed"]
        density[i, j] = row["density"]
    figure, axes = plt.subplots(2, 1, figsize=(8.0, 6.5), sharex=True)
    for axis, values, title, cmap in (
        (axes[0], speed, "x-t speed", "viridis"),
        (axes[1], density, "x-t density", "magma"),
    ):
        image = axis.imshow(values, aspect="auto", origin="lower", cmap=cmap, extent=[min(times), max(times), min(positions), max(positions)])
        axis.set_ylabel("Route edge position")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, fraction=0.025)
    axes[-1].set_xlabel("Simulation time (s)")
    figure.suptitle(f"{case_id}: adaptive spatiotemporal mechanism")
    figure.tight_layout()
    heatmap = directory / f"{case_id}_xt_heatmap.png"
    figure.savefig(heatmap, dpi=170)
    plt.close(figure)
    outputs.append(str(heatmap.relative_to(ROOT)))
    return outputs


def run() -> Path:
    verify_frozen()
    rows = json.loads((SOURCE / "raw_metrics.json").read_text())
    decisions = json.loads((SOURCE / "decision_log.json").read_text())
    indexed_decisions = {row["case_id"]: row for row in decisions}
    failed_interventions = [
        row
        for row in rows
        if indexed_decisions[row["case_id"]]["intervene"]
        and not row["label"]["safe_micro_success"]
    ]
    analysis_basis = "executed_v6_false_positives"
    if not failed_interventions:
        failed_interventions = sorted(
            [row for row in rows if not row["label"]["safe_micro_success"]],
            key=lambda row: indexed_decisions[row["case_id"]]["composite_probability"],
            reverse=True,
        )[:3]
        analysis_basis = "highest_score_counterfactual_B6_failures_under_safe_abstention"
    else:
        failed_interventions = failed_interventions[:3]
    config = yaml.safe_load((ROOT / "configs/v6/micro_design.yaml").read_text())
    attribution = []
    with tempfile.TemporaryDirectory(prefix="concordia-v6-failure-") as temporary:
        directory = Path(temporary)
        networks = {}
        for row in failed_interventions:
            condition = row["condition"]
            key = (condition["topology"], condition["perturbation"])
            if key not in networks:
                networks[key] = build_v6_network(directory, *key)
            network, metadata = networks[key]
            analytical = {
                "success_probability": row["features_pre_decision"]["analytical_success_probability"],
                "predicted_ttt_gain": row["features_pre_decision"]["analytical_predicted_ttt_gain"],
            }
            baseline, adaptive = run_v6_pair(
                network, metadata, config, condition, int(row["seed"]), analytical,
                capture_diagnostics=True,
            )
            decision = indexed_decisions[row["case_id"]]
            attribution.append(
                {
                    "case_id": row["case_id"],
                    "executed_by_v6": decision["intervene"],
                    "predicted_safe_success_probability": decision["composite_probability"],
                    "realized_safe_success": row["label"]["safe_micro_success"],
                    "traffic_gain": row["label"]["relative_ttt_gain"],
                    "safety_violation": not row["label"]["safety_pass"],
                    "penetration": row["condition"]["penetration"],
                    "failure_type": _failure_type(row),
                    "figures": _plot(row["case_id"], baseline, adaptive),
                }
            )
    summary = {
        "complete": True,
        "study": "v6 microscopic failure mechanism analysis",
        "analysis_basis": analysis_basis,
        "executed_false_positive_count": sum(
            decision["intervene"] and not row["label"]["safe_micro_success"]
            for row, decision in zip(rows, decisions)
        ),
        "visualized_case_count": len(attribution),
        "failure_taxonomy": dict(Counter(row["failure_type"] for row in attribution)),
        "thresholds_changed": False,
        "full_flow_speed_queue_drac_ttc_series": True,
        "route_load_plots": True,
        "spatiotemporal_speed_density_heatmaps": True,
    }
    write_json(OUTPUT / "failure_attribution.json", attribution)
    write_json(OUTPUT / "summary.json", summary)
    print(OUTPUT / "summary.json")
    return OUTPUT / "summary.json"


if __name__ == "__main__":
    run()
