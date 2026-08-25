#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/concordia-matplotlib-v3")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/concordia-cache-v3")

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from concordia.evaluation import ExperimentRegistry, capture_source_state, summarize_selective_policy
from concordia.feasibility import BootstrapFeasibilityEnsemble, FEATURE_SCHEMA, FeasibilityGate
from concordia.feasibility.dataset import build_alignment_case
from concordia.selective import SelectiveInterventionPolicy


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/v3/tail_study.yaml"
MODEL_STUDY = ROOT / "artifacts/studies/v3_feasibility_prediction"
HOLDOUT = ROOT / "artifacts/studies/v3_selective_holdout"
OUTPUT = ROOT / "artifacts/studies/v3_tail_robustness"


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _regression_predict(package: dict, matrix: np.ndarray) -> np.ndarray:
    mean = np.asarray(package["mean"], dtype=float)
    scale = np.asarray(package["scale"], dtype=float)
    coefficients = np.asarray(package["coefficients"], dtype=float)
    return package["intercept"] + ((matrix - mean) / scale) @ coefficients


def run() -> Path:
    source_commit, source_dirty = capture_source_state()
    started = datetime.now(timezone.utc)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    frozen_path = ROOT / "configs/v3/frozen_thresholds.yaml"
    selected_path = MODEL_STUDY / "selected_model.json"
    frozen_hash_before = _sha(frozen_path)
    frozen = yaml.safe_load(frozen_path.read_text(encoding="utf-8"))
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    if _sha(selected_path) != frozen["selected_model_hash"]:
        raise RuntimeError("selected feasibility model changed after freeze")
    base_rows = json.loads((HOLDOUT / "raw_metrics.json").read_text(encoding="utf-8"))
    base_decisions = json.loads((HOLDOUT / "decision_log.json").read_text(encoding="utf-8"))
    if len(base_rows) != len(base_decisions):
        raise RuntimeError("holdout raw metrics and decision log are misaligned")
    ensemble = BootstrapFeasibilityEnsemble.from_dict(selected["ensemble"])
    policy = SelectiveInterventionPolicy(
        FeasibilityGate(
            probability_threshold=float(frozen["p_win_threshold"]),
            maximum_uncertainty=float(frozen["maximum_uncertainty"]),
            safety_delta=float(frozen["safety_delta"]),
            minimum_acceptance_probability=float(frozen["minimum_acceptance_probability"]),
            minimum_ttt_lcb_gain=float(frozen["minimum_relative_ttt_gain"]),
            maximum_tail_loss=float(frozen["maximum_tail_loss"]),
        )
    )
    stress_rows = []
    for base in base_rows:
        stress = build_alignment_case(
            scenario=base["scenario"],
            seed=int(base["seed"]),
            demand_scale=float(base["condition"]["demand_scale"])
            * float(config["demand_shift_multiplier"]),
            heterogeneity=base["condition"]["heterogeneity"],
            navigation_penetration=float(base["condition"]["navigation_penetration"]),
            user_count=6,
            regret_limit=0.08,
            epsilon_grid=[0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.24],
            minimum_relative_ttt_gain=float(frozen["minimum_relative_ttt_gain"]),
            safety_delta=float(frozen["safety_delta"]),
            source_split="post_holdout_stress",
        )
        stress["base_holdout_case_id"] = base["case_id"]
        stress["features"]["preference_variance"] *= float(
            config["preference_variance_multiplier"]
        )
        stress["features"]["heterogeneity_rad_interaction"] = (
            stress["features"]["preference_variance"]
            * stress["features"]["route_attribute_diversity"]
        )
        stress_rows.append(stress)
    matrix = np.asarray(
        [[row["features"][name] for name in FEATURE_SCHEMA] for row in stress_rows],
        dtype=float,
    )
    probability, uncertainty, lower = ensemble.predict(matrix)
    benefit = _regression_predict(selected["benefit_regression"], matrix)
    benefit_lcb = benefit - frozen["benefit_lcb_z"] * selected["benefit_regression"][
        "residual_standard_deviation"
    ]
    safety = _regression_predict(selected["safety_regression"], matrix)
    safety_upper = safety + frozen["benefit_lcb_z"] * selected["safety_regression"][
        "residual_standard_deviation"
    ]
    decisions = []
    policy_rows = []
    base_relative_gain = []
    stress_relative_gain = []
    for index, row in enumerate(stress_rows):
        features = row["features"]
        decision = policy.decide(
            case_id=row["case_id"],
            p_win=float(probability[index]),
            p_win_lower=float(lower[index]),
            uncertainty=float(uncertainty[index]),
            alignment_potential=features["alignment_potential_score"],
            route_overlap=features["route_overlap"],
            safety_upper_difference=float(safety_upper[index]),
            acceptance_probability=features["acceptance_probability"],
            ttt_lcb_gain=float(benefit_lcb[index]),
            predicted_tail_loss=max(0.0, -float(benefit_lcb[index])),
            legal=bool(row["adaptive_counterfactual"]["legal"]),
        )
        baseline_ttt = float(row["baseline_metrics"]["eta_only_ttt"])
        adaptive_ttt = float(row["adaptive_counterfactual"]["ttt"])
        selected_ttt = adaptive_ttt if decision.intervene else baseline_ttt
        success = row["label"] == "WIN"
        policy_rows.append(
            {
                "case_id": row["case_id"],
                "intervene": decision.intervene,
                "success": bool(decision.intervene and success),
                "counterfactual_success": success,
                "system_ttt_gain": baseline_ttt - selected_ttt,
                "baseline_ttt": baseline_ttt,
                "selected_ttt": selected_ttt,
                "regret_violation": row["adaptive_counterfactual"]["maximum_regret"] > 0.08 + 1e-10,
                "safety_violation": row["adaptive_counterfactual"]["safety_difference"]
                > frozen["safety_delta"] + 1e-10,
                "legal_violation": not bool(row["adaptive_counterfactual"]["legal"]),
            }
        )
        base = base_rows[index]
        base_baseline = float(base["baseline_metrics"]["eta_only_ttt"])
        base_adaptive = float(base["adaptive_counterfactual"]["ttt"])
        base_intervene = bool(base_decisions[index]["intervene"])
        base_relative_gain.append(
            (base_baseline - (base_adaptive if base_intervene else base_baseline))
            / max(base_baseline, 1e-9)
        )
        stress_relative_gain.append((baseline_ttt - selected_ttt) / max(baseline_ttt, 1e-9))
        decisions.append(
            {
                **decision.__dict__,
                "outcome": "SUCCESS" if decision.intervene and success else "FAILURE" if decision.intervene else "ABSTAIN",
                "base_holdout_case_id": row["base_holdout_case_id"],
                "threshold_hash": frozen_hash_before,
            }
        )
    metrics = summarize_selective_policy(policy_rows)
    degradation = np.asarray(base_relative_gain) - np.asarray(stress_relative_gain)
    quantile = float(config["cvar_quantile"])
    cutoff = float(np.quantile(degradation, quantile))
    tail = degradation[degradation >= cutoff]
    cvar = float(tail.mean()) if len(tail) else 0.0
    frozen_hash_after = _sha(frozen_path)
    if frozen_hash_after != frozen_hash_before:
        raise RuntimeError("frozen threshold configuration changed during stress study")
    statistics = {
        "H10_stress_mean_network_gain_nonnegative": metrics["mean_network_ttt_gain"] >= 0.0,
        "H10_stress_safety_violation_count": metrics["safety_violation_count"],
        "H10_tail_degradation_quantile": quantile,
        "H10_tail_degradation_var": cutoff,
        "H10_tail_degradation_cvar": cvar,
        "H10_tail_limit": float(config["maximum_tail_loss"]),
        "H10_tail_gate_pass": cvar <= float(config["maximum_tail_loss"]),
        "mean_relative_gain_base_holdout": float(np.mean(base_relative_gain)),
        "mean_relative_gain_stress": float(np.mean(stress_relative_gain)),
    }
    summary = {
        "complete": True,
        "study": "Study IX — Tail Robustness",
        "post_holdout": True,
        "threshold_immutable": True,
        "threshold_hash_before": frozen_hash_before,
        "threshold_hash_after": frozen_hash_after,
        "demand_shift_multiplier": config["demand_shift_multiplier"],
        "preference_variance_multiplier": config["preference_variance_multiplier"],
        "preference_shift_scope": "model-input covariate sensitivity; realized counterfactual uses demand-shifted synthetic population",
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
    axis.hist(degradation, bins=24, color="#555555")
    axis.axvline(cutoff, linestyle="--", color="#b51f1f", label=f"VaR {quantile:.2f}")
    axis.set(xlabel="Relative-gain degradation", ylabel="Cases", title="Post-holdout stress tail")
    axis.legend()
    fig.tight_layout()
    figure_path = figure_dir / "tail_degradation.png"
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
    run_dir = ExperimentRegistry(str(ROOT / "artifacts/runs")).create(
        config,
        summary,
        input_paths=(
            str(CONFIG.relative_to(ROOT)),
            "configs/v3/frozen_thresholds.yaml",
            str(selected_path.relative_to(ROOT)),
            str((HOLDOUT / "summary.json").relative_to(ROOT)),
        ),
        external_output_paths=tuple(str(path.relative_to(ROOT)) for path in outputs),
        started_at=started,
        ended_at=ended,
        source_commit=source_commit,
        source_dirty=source_dirty,
    )
    shutil.copyfile(run_dir / "manifest.json", OUTPUT / "manifest.json")
    _write(
        OUTPUT / "v3_registry.json",
        {
            "git_commit": source_commit,
            "git_dirty": source_dirty,
            "model_hash": _sha(selected_path),
            "threshold_config_hash": frozen_hash_before,
            "result_hash": _sha(summary_path),
        },
    )
    print(summary_path)
    return summary_path


if __name__ == "__main__":
    run()
