#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import shutil
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


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/v4/stress.yaml"
VALIDATION = ROOT / "artifacts/studies/v4_precision_validation"
HOLDOUT = ROOT / "artifacts/studies/v4_frozen_holdout"
OUTPUT = ROOT / "artifacts/studies/v4_stress"


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


def _stress_features(case: dict, config: dict) -> dict[str, float]:
    features = expand_v4_features(case)
    features["preference_variance"] *= float(
        config["preference_variance_multiplier"]
    )
    features["acceptance_probability"] = float(
        np.clip(
            features["acceptance_probability"]
            * float(config["acceptance_probability_multiplier"]),
            0.0,
            1.0,
        )
    )
    features["heterogeneity_rad_interaction"] = (
        features["preference_variance"]
        * features["route_attribute_diversity"]
    )
    return {name: float(features[name]) for name in V4_FEATURE_SCHEMA}


def run() -> Path:
    source_commit, source_dirty = capture_source_state()
    started = datetime.now(timezone.utc)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    prereg = yaml.safe_load(
        (ROOT / "configs/v4/preregistration.yaml").read_text(encoding="utf-8")
    )
    frozen_model_path = ROOT / "configs/v4/frozen_model.yaml"
    frozen_threshold_path = ROOT / "configs/v4/frozen_thresholds.yaml"
    model_hash = _sha(frozen_model_path)
    threshold_hash = _sha(frozen_threshold_path)
    probability = json.loads(
        (VALIDATION / "probability_package.json").read_text(encoding="utf-8")
    )
    benefit = json.loads(
        (VALIDATION / "benefit_package.json").read_text(encoding="utf-8")
    )
    safety = json.loads(
        (VALIDATION / "safety_package.json").read_text(encoding="utf-8")
    )
    selection = json.loads(
        (VALIDATION / "threshold_selection.json").read_text(encoding="utf-8")
    )
    bundle = V4PredictionBundle.from_packages(probability, benefit, safety)
    policy = _policy(selection)
    base_rows = json.loads((HOLDOUT / "raw_metrics.json").read_text(encoding="utf-8"))
    base_policy = json.loads(
        (HOLDOUT / "policy_rows.json").read_text(encoding="utf-8")
    )["V4-F"]
    if len(base_rows) != len(base_policy):
        raise RuntimeError("v4 holdout rows and policy decisions are misaligned")

    stress_rows = []
    base_selected_gain: dict[str, float] = {}
    for base, selected in zip(base_rows, base_policy):
        base_selected_gain[base["case_id"]] = float(selected["relative_ttt_gain"])
        for multiplier in config["demand_multipliers"]:
            case = build_alignment_case(
                scenario=str(base["scenario"]),
                seed=int(base["seed"]),
                demand_scale=float(base["condition"]["demand_scale"]) * float(multiplier),
                heterogeneity=str(base["condition"]["heterogeneity"]),
                navigation_penetration=float(
                    base["condition"]["navigation_penetration"]
                ),
                user_count=6,
                regret_limit=float(prereg["success_definition"]["regret_limit"]),
                epsilon_grid=[0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.24],
                minimum_relative_ttt_gain=float(
                    prereg["success_definition"]["minimum_relative_ttt_gain"]
                ),
                safety_delta=float(prereg["success_definition"]["safety_delta"]),
                source_split=f"v4_stress_demand_{multiplier}",
            )
            case["base_holdout_case_id"] = base["case_id"]
            case["demand_multiplier"] = float(multiplier)
            case["features"] = _stress_features(case, config)
            stress_rows.append(case)

    matrix = np.asarray(
        [[row["features"][name] for name in V4_FEATURE_SCHEMA] for row in stress_rows],
        dtype=float,
    )
    prediction = bundle.predict(matrix)
    decisions = []
    policy_rows = []
    losses = []
    degradations = []
    for index, row in enumerate(stress_rows):
        inputs = V4DecisionInputs(
            case_id=row["case_id"],
            success_probability=float(prediction["probability"][index]),
            success_probability_lower=float(prediction["probability_lower"][index]),
            expected_benefit=float(prediction["expected_benefit"][index]),
            benefit_lower=float(prediction["benefit_lower"][index]),
            safety_difference_upper=float(prediction["safety_difference_upper"][index]),
            safety_failure_probability=float(
                prediction["safety_failure_probability"][index]
            ),
            safety_failure_probability_upper=float(
                prediction["safety_failure_probability_upper"][index]
            ),
            esiv=float(prediction["esiv"][index]),
            esiv_lower=float(prediction["esiv_lower"][index]),
            legal=bool(row["adaptive_counterfactual"]["legal"]),
        )
        decision = policy.decide(inputs)
        baseline_ttt = float(row["baseline_metrics"]["eta_only_ttt"])
        adaptive_ttt = float(row["adaptive_counterfactual"]["ttt"])
        selected_ttt = adaptive_ttt if decision.intervene else baseline_ttt
        relative_gain = (baseline_ttt - selected_ttt) / max(baseline_ttt, 1e-9)
        success = row["label"] == "WIN"
        policy_rows.append(
            {
                "case_id": row["case_id"],
                "intervene": decision.intervene,
                "success": bool(decision.intervene and success),
                "counterfactual_success": success,
                "system_ttt_gain": baseline_ttt - selected_ttt,
                "relative_ttt_gain": relative_gain,
                "regret_violation": row["adaptive_counterfactual"]["maximum_regret"]
                > prereg["success_definition"]["regret_limit"] + 1e-10,
                "safety_violation": row["adaptive_counterfactual"]["safety_difference"]
                > prereg["success_definition"]["safety_delta"] + 1e-10,
                "legal_violation": not bool(row["adaptive_counterfactual"]["legal"]),
                "demand_multiplier": row["demand_multiplier"],
            }
        )
        loss = max(0.0, -relative_gain)
        degradation = max(
            0.0,
            base_selected_gain[row["base_holdout_case_id"]] - relative_gain,
        )
        losses.append(loss)
        degradations.append(degradation)
        decisions.append(
            {
                **decision.__dict__,
                "outcome": "SUCCESS" if decision.intervene and success else "FAILURE" if decision.intervene else "ABSTAIN",
                "base_holdout_case_id": row["base_holdout_case_id"],
                "demand_multiplier": row["demand_multiplier"],
                "realized_relative_ttt_gain": relative_gain,
                "model_hash": model_hash,
                "threshold_hash": threshold_hash,
            }
        )

    metrics = summarize_selective_policy(policy_rows)
    quantile = float(config["cvar_quantile"])
    loss_cutoff = float(np.quantile(losses, quantile))
    loss_tail = [value for value in losses if value >= loss_cutoff]
    degradation_cutoff = float(np.quantile(degradations, quantile))
    degradation_tail = [value for value in degradations if value >= degradation_cutoff]
    statistics = {
        "precision_stress": metrics["intervention_precision"],
        "coverage_stress": metrics["coverage"],
        "stress_precision_target": float(config["stress_precision_target"]),
        "stress_precision_target_met": metrics["intervention_precision"]
        >= float(config["stress_precision_target"]),
        "safety_violation_count": metrics["safety_violation_count"],
        "cvar_quantile": quantile,
        "loss_var": loss_cutoff,
        "loss_cvar": float(np.mean(loss_tail)) if loss_tail else 0.0,
        "degradation_var": degradation_cutoff,
        "degradation_cvar": float(np.mean(degradation_tail))
        if degradation_tail
        else 0.0,
        "tail_limit": float(config["maximum_tail_loss"]),
        "tail_limit_met": (
            float(np.mean(degradation_tail)) if degradation_tail else 0.0
        )
        <= float(config["maximum_tail_loss"]),
    }
    if _sha(frozen_model_path) != model_hash or _sha(frozen_threshold_path) != threshold_hash:
        raise RuntimeError("frozen v4 artifacts changed during stress evaluation")
    summary = {
        "complete": True,
        "study": "Study XV — Stress / Distribution Shift",
        "post_holdout": True,
        "frozen_immutable": True,
        "case_count": len(stress_rows),
        "demand_multipliers": config["demand_multipliers"],
        "preference_variance_multiplier": config[
            "preference_variance_multiplier"
        ],
        "acceptance_probability_multiplier": config[
            "acceptance_probability_multiplier"
        ],
        "policy_metrics": metrics,
        "statistics": statistics,
        "claim_boundary": config["claim_boundary"],
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    raw_path = OUTPUT / "raw_metrics.json"
    processed_path = OUTPUT / "processed_metrics.json"
    statistics_path = OUTPUT / "statistical_tests.json"
    summary_path = OUTPUT / "summary.json"
    decision_path = OUTPUT / "decision_log.json"
    _write(raw_path, stress_rows)
    _write(processed_path, metrics)
    _write(statistics_path, statistics)
    _write(summary_path, summary)
    _write(decision_path, decisions)
    figure_dir = OUTPUT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.hist(degradations, bins=28, color="#3e6f61")
    axis.axvline(
        degradation_cutoff,
        linestyle="--",
        color="#9b2f2f",
        label=f"VaR {quantile:.0%}",
    )
    axis.set(
        xlabel="Relative-gain degradation",
        ylabel="Cases",
        title="Frozen v4 distribution-shift tail",
    )
    axis.legend()
    fig.tight_layout()
    figure_path = figure_dir / "stress_tail.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    ended = datetime.now(timezone.utc)
    outputs = (
        raw_path,
        processed_path,
        statistics_path,
        summary_path,
        decision_path,
        figure_path,
    )
    registry = ExperimentRegistry(str(ROOT / "artifacts/runs")).create(
        config,
        summary,
        input_paths=(
            "configs/v4/stress.yaml",
            "configs/v4/frozen_model.yaml",
            "configs/v4/frozen_thresholds.yaml",
            "artifacts/studies/v4_frozen_holdout/summary.json",
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
