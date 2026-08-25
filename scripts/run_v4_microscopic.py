#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/concordia-matplotlib-v4")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/concordia-cache-v4")

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from concordia.evaluation import ExperimentRegistry, capture_source_state, summarize_selective_policy
from concordia.feasibility import (
    V4PredictionBundle,
    V4_FEATURE_SCHEMA,
    build_alignment_case,
    expand_v4_features,
)
from concordia.selective import PrecisionConstrainedPolicy, V4DecisionInputs
from concordia.simulation import SumoAdapter
from run_microscopic_study import _build_network, _run_one


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/v4/microscopic.yaml"
BASE_CONFIG = ROOT / "configs/experiments/microscopic_policy_matrix.yaml"
VALIDATION = ROOT / "artifacts/studies/v4_precision_validation"
OUTPUT = ROOT / "artifacts/studies/v4_microscopic"


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _policy(selection: dict) -> PrecisionConstrainedPolicy:
    name = selection["selected_policy"]
    point = selection["policy_operating_points"][name]
    return PrecisionConstrainedPolicy(
        name,
        probability_threshold=float(point["score_threshold"]) if name != "V4-E" else 0.0,
        benefit_threshold=float(point["benefit_threshold"]),
        safety_delta=float(selection["safety_delta"]),
        safety_probability_threshold=float(
            selection["safety_failure_probability_threshold"]
        ),
        esiv_threshold=float(point["score_threshold"]) if name == "V4-E" else 0.0,
        use_esiv=name == "V4-E",
    )


def run() -> Path:
    source_commit, source_dirty = capture_source_state()
    started = datetime.now(timezone.utc)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    base = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    frozen_model = ROOT / "configs/v4/frozen_model.yaml"
    frozen_thresholds = ROOT / "configs/v4/frozen_thresholds.yaml"
    model_hash = _sha(frozen_model)
    threshold_hash = _sha(frozen_thresholds)
    probability = json.loads(
        (VALIDATION / "probability_package.json").read_text(encoding="utf-8")
    )
    benefit = json.loads((VALIDATION / "benefit_package.json").read_text(encoding="utf-8"))
    safety = json.loads((VALIDATION / "safety_package.json").read_text(encoding="utf-8"))
    selection = json.loads(
        (VALIDATION / "threshold_selection.json").read_text(encoding="utf-8")
    )
    bundle = V4PredictionBundle.from_packages(probability, benefit, safety)
    policy = _policy(selection)
    conditions = []
    for seed in config["seeds"]:
        for demand in (800, 1200):
            heterogeneities = ("low", "high") if demand == 800 else ("high",)
            for heterogeneity in heterogeneities:
                conditions.append((int(seed), demand, 0.5, heterogeneity, "potentially_beneficial"))
        for heterogeneity in ("low", "high"):
            conditions.append((int(seed), 1600, 1.0, heterogeneity, "potentially_risky"))
    decisions = {}
    decision_log = []
    for seed, demand, penetration, heterogeneity, regime in conditions:
        analytical = build_alignment_case(
            scenario="merge",
            seed=seed,
            demand_scale=demand / 1200.0,
            heterogeneity=heterogeneity,
            navigation_penetration=penetration,
            user_count=6,
            regret_limit=0.08,
            epsilon_grid=[0.0, 0.02, 0.04, 0.08, 0.12, 0.16],
            minimum_relative_ttt_gain=0.01,
            safety_delta=0.25,
            source_split="v4_microscopic_pre_run",
        )
        features = expand_v4_features(analytical)
        matrix = np.asarray([[features[name] for name in V4_FEATURE_SCHEMA]], dtype=float)
        prediction = bundle.predict(matrix)
        inputs = V4DecisionInputs(
            case_id=f"micro-s{seed}-d{demand}-p{penetration}-{heterogeneity}",
            success_probability=float(prediction["probability"][0]),
            success_probability_lower=float(prediction["probability_lower"][0]),
            expected_benefit=float(prediction["expected_benefit"][0]),
            benefit_lower=float(prediction["benefit_lower"][0]),
            safety_difference_upper=float(prediction["safety_difference_upper"][0]),
            safety_failure_probability=float(prediction["safety_failure_probability"][0]),
            safety_failure_probability_upper=float(
                prediction["safety_failure_probability_upper"][0]
            ),
            esiv=float(prediction["esiv"][0]),
            esiv_lower=float(prediction["esiv_lower"][0]),
            legal=True,
        )
        decision = policy.decide(inputs)
        key = (seed, demand, penetration, heterogeneity)
        decisions[key] = decision
        decision_log.append(
            {
                **decision.__dict__,
                "outcome": "PENDING_ACTUAL_SUMO" if decision.intervene else "ABSTAIN",
                "regime": regime,
                "model_hash": model_hash,
                "threshold_hash": threshold_hash,
            }
        )
    with tempfile.TemporaryDirectory(prefix="concordia-v4-micro-") as temporary:
        network = _build_network(Path(temporary))
        actual = []
        for seed, demand, penetration, heterogeneity, regime in conditions:
            for policy_name in ("B1", "B6"):
                row = _run_one(
                    network,
                    base,
                    policy_name,
                    seed,
                    demand,
                    penetration,
                    heterogeneity,
                )
                row["regime"] = regime
                actual.append(row)
    pairs = defaultdict(dict)
    for row in actual:
        key = (
            row["seed"],
            row["demand_vehicles_per_hour"],
            row["navigation_penetration"],
            row["heterogeneity"],
        )
        pairs[key][row["policy"]] = row
    policy_rows = {"B6": [], "V4-F": []}
    selected_actual = []
    for key, pair in sorted(pairs.items()):
        decision = decisions[key]
        b1, b6 = pair["B1"], pair["B6"]
        relative_gain = (b1["total_travel_time_seconds"] - b6["total_travel_time_seconds"]) / max(
            b1["total_travel_time_seconds"], 1e-9
        )
        safety_difference = b6["safety"]["cvar_drac_95"] - b1["safety"]["cvar_drac_95"]
        success = relative_gain >= 0.01 and safety_difference <= 0.25
        selected_actual.append(
            {
                **(b6 if decision.intervene else b1),
                "policy": "V4-F",
                "pre_run_intervene": decision.intervene,
            }
        )
        for name, intervene in (("B6", True), ("V4-F", decision.intervene)):
            selected_ttt = b6["total_travel_time_seconds"] if intervene else b1[
                "total_travel_time_seconds"
            ]
            policy_rows[name].append(
                {
                    "case_id": decision.case_id,
                    "intervene": intervene,
                    "success": bool(intervene and success),
                    "counterfactual_success": success,
                    "system_ttt_gain": b1["total_travel_time_seconds"] - selected_ttt,
                    "regret_violation": False,
                    "safety_violation": safety_difference > 0.25,
                    "legal_violation": False,
                    "regime": b1["regime"],
                }
            )
    metrics = {name: summarize_selective_policy(values) for name, values in policy_rows.items()}
    regime_metrics = {}
    for regime in ("potentially_beneficial", "potentially_risky"):
        selected = [row for row in policy_rows["V4-F"] if row["regime"] == regime]
        regime_metrics[regime] = summarize_selective_policy(selected)
    intervention_count = metrics["V4-F"]["intervention_count"]
    summary = {
        "complete": True,
        "study": "Study XIII — Microscopic v4",
        "simulator_version": SumoAdapter.simulator_version(),
        "pair_count": len(pairs),
        "policy_metrics": metrics,
        "regime_metrics": regime_metrics,
        "microscopic_interventions_positive": intervention_count > 0,
        "adaptive_success_claim_allowed": intervention_count > 0,
        "safety_abstention_claim_allowed": metrics["V4-F"]["safety_violation_count"] == 0,
        "phantom_role": "secondary event analysis only; no phantom predictor in V4-F",
        "claim_boundary": config["claim_boundary"],
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    raw_path = OUTPUT / "raw_metrics.json"
    processed_path = OUTPUT / "processed_metrics.json"
    summary_path = OUTPUT / "summary.json"
    decision_path = OUTPUT / "decision_log.json"
    _write(raw_path, actual + selected_actual)
    _write(processed_path, {"policy": policy_rows, "metrics": metrics})
    _write(summary_path, summary)
    _write(decision_path, decision_log)
    figure_dir = OUTPUT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(6.2, 4.2))
    axis.bar(
        ["B6 precision", "V4 precision", "V4 coverage"],
        [
            metrics["B6"]["intervention_precision"],
            metrics["V4-F"]["intervention_precision"],
            metrics["V4-F"]["coverage"],
        ],
        color=["#555555", "#0b6e4f", "#79aa98"],
    )
    axis.set_ylim(0, 1)
    fig.tight_layout()
    figure_path = figure_dir / "microscopic_v4_selectivity.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    ended = datetime.now(timezone.utc)
    outputs = (raw_path, processed_path, summary_path, decision_path, figure_path)
    registry = ExperimentRegistry(str(ROOT / "artifacts/runs")).create(
        config,
        summary,
        simulator_version=SumoAdapter.simulator_version(),
        input_paths=(
            "configs/v4/microscopic.yaml",
            "configs/v4/frozen_model.yaml",
            "configs/v4/frozen_thresholds.yaml",
        ),
        external_output_paths=tuple(str(path.relative_to(ROOT)) for path in outputs),
        started_at=started,
        ended_at=ended,
        source_commit=source_commit,
        source_dirty=source_dirty,
    )
    shutil.copyfile(registry / "manifest.json", OUTPUT / "manifest.json")
    _write(
        OUTPUT / "v4_registry.json",
        {
            "git_commit": source_commit,
            "git_dirty": source_dirty,
            "model_hash": model_hash,
            "threshold_hash": threshold_hash,
            "result_hash": _sha(summary_path),
        },
    )
    print(summary_path)
    return summary_path


if __name__ == "__main__":
    run()
