#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import time
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
from concordia.optimization.rl0 import PPOEligibilityPolicy


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiments" / "conditional_rl.yaml"
OUTPUT = ROOT / "artifacts" / "studies" / "conditional_rl"


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _state(row: dict, scenarios: list[str]) -> list[float]:
    private = float(row["private_best_ttt"])
    eta = float(row["eta_only_ttt"])
    demand = float(row["demand_scale"])
    speed = eta / max(private, 1e-9)
    density = demand / max(speed, 1e-6)
    occupancy = min(1.0, density / 2.5)
    phantom_risk = 1.0 / (1.0 + np.exp(-(density - speed)))
    heterogeneity = {"low": 0.0, "medium": 0.5, "high": 1.0}.get(
        str(row["heterogeneity"]), 0.5
    )
    return [
        demand,
        density,
        speed,
        occupancy,
        float(phantom_risk),
        demand,
        heterogeneity,
        private / max(eta, 1e-9),
        float(row["frontier"][0]["safety_risk"]),
        *[float(row["scenario"] == scenario) for scenario in scenarios],
    ]


def _action_tables(rows: list[dict], config: dict) -> tuple[np.ndarray, np.ndarray]:
    actions = [float(value) for value in config["actions"]["preference_epsilon"]]
    rewards = np.zeros((len(rows), len(actions)), dtype=float)
    valid = np.zeros_like(rewards, dtype=bool)
    for row_index, row in enumerate(rows):
        points = {float(point["epsilon"]): point for point in row["frontier"]}
        baseline_safety = float(points[actions[0]]["safety_risk"])
        for action_index, epsilon in enumerate(actions):
            point = points[epsilon]
            valid[row_index, action_index] = (
                point["max_regret"] <= epsilon + 1e-10
                and point["safety_risk"]
                <= baseline_safety + float(config["safety_delta"]) + 1e-10
            )
            rewards[row_index, action_index] = -(
                float(config["reward"]["normalized_ttt_weight"])
                * point["minimum_feasible_ttt"]
                / row["unconstrained_system_optimum_ttt"]
                + float(config["reward"]["nonacceptance_weight"])
                * (1.0 - point["acceptance_rate"])
            )
            if not valid[row_index, action_index]:
                rewards[row_index, action_index] = -float(
                    config["reward"]["invalid_action_penalty"]
                )
    return rewards, valid


def run() -> Path:
    source_commit, source_dirty = capture_source_state()
    started = datetime.now(timezone.utc)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    source_path = ROOT / config["source_study"]
    rows = json.loads(source_path.read_text(encoding="utf-8"))
    train = [row for row in rows if row["seed"] in config["train_seeds"]]
    test = [row for row in rows if row["seed"] in config["test_seeds"]]
    scenarios = sorted({str(row["scenario"]) for row in rows})
    train_states = np.asarray([_state(row, scenarios) for row in train], dtype=float)
    test_states = np.asarray([_state(row, scenarios) for row in test], dtype=float)
    mean = train_states.mean(axis=0)
    scale = train_states.std(axis=0)
    scale[scale < 1e-9] = 1.0
    train_states = (train_states - mean) / scale
    test_states = (test_states - mean) / scale
    train_rewards, train_valid = _action_tables(train, config)
    test_rewards, test_valid = _action_tables(test, config)
    ppo = config["ppo"]
    policy = PPOEligibilityPolicy(
        train_states.shape[1], train_rewards.shape[1], int(config["policy_seed"])
    )
    telemetry = policy.fit(
        train_states,
        train_rewards,
        train_valid,
        epochs=int(ppo["epochs"]),
        update_epochs=int(ppo["update_epochs"]),
        learning_rate=float(ppo["learning_rate"]),
        value_learning_rate=float(ppo["value_learning_rate"]),
        clip_ratio=float(ppo["clip_ratio"]),
        entropy_coefficient=float(ppo["entropy_coefficient"]),
    )
    inference_times = []
    rl_actions = []
    for index in range(len(test)):
        tick = time.perf_counter_ns()
        action = int(policy.act(test_states[index : index + 1], test_valid[index : index + 1])[0])
        inference_times.append((time.perf_counter_ns() - tick) / 1e9)
        rl_actions.append(action)
    deterministic_actions = np.argmax(test_rewards, axis=1)
    actions = [float(value) for value in config["actions"]["preference_epsilon"]]
    fixed_b6_index = actions.index(0.08)
    raw = []
    for index, row in enumerate(test):
        points = row["frontier"]
        rl_point = points[rl_actions[index]]
        deterministic_point = points[int(deterministic_actions[index])]
        b6_point = points[fixed_b6_index]
        raw.append(
            {
                "scenario": row["scenario"],
                "seed": row["seed"],
                "demand_scale": row["demand_scale"],
                "heterogeneity": row["heterogeneity"],
                "rl_action_preference_epsilon": actions[rl_actions[index]],
                "rl_ttt": rl_point["minimum_feasible_ttt"],
                "deterministic_approximation_ttt": deterministic_point[
                    "minimum_feasible_ttt"
                ],
                "fixed_b6_epsilon_ttt": b6_point["minimum_feasible_ttt"],
                "rl_gap_vs_deterministic": (
                    rl_point["minimum_feasible_ttt"]
                    - deterministic_point["minimum_feasible_ttt"]
                )
                / deterministic_point["minimum_feasible_ttt"],
                "regret_violation": rl_point["max_regret"]
                > actions[rl_actions[index]] + 1e-10,
                "safety_violation": not bool(test_valid[index, rl_actions[index]]),
                "inference_seconds": inference_times[index],
            }
        )
    rl_ttt = [row["rl_ttt"] for row in raw]
    deterministic_ttt = [row["deterministic_approximation_ttt"] for row in raw]
    b6_ttt = [row["fixed_b6_epsilon_ttt"] for row in raw]
    mean_gap = float(np.mean([row["rl_gap_vs_deterministic"] for row in raw]))
    p95_inference = float(np.percentile(inference_times, 95))
    regret_violations = sum(row["regret_violation"] for row in raw)
    safety_violations = sum(row["safety_violation"] for row in raw)
    nonstationarity_superior = float(np.mean(rl_ttt)) < float(np.mean(deterministic_ttt)) - 1e-9
    retention = config["retention"]
    retained = (
        mean_gap <= float(retention["maximum_mean_ttt_gap_vs_deterministic"])
        and p95_inference <= float(retention["maximum_p95_inference_seconds"])
        and (not retention["require_zero_regret_violations"] or regret_violations == 0)
        and (not retention["require_zero_safety_violations"] or safety_violations == 0)
        and (not retention["require_nonstationarity_superiority"] or nonstationarity_superior)
    )
    statistics = {
        "RL0_vs_deterministic": paired_comparison(deterministic_ttt, rl_ttt, seed=79),
        "RL0_vs_fixed_B6": paired_comparison(b6_ttt, rl_ttt, seed=83),
    }
    summary = {
        "complete": True,
        "study": "Conditional RL0 — compact PPO eligibility policy",
        "algorithm": "masked clipped PPO with linear actor/critic",
        "train_case_count": len(train),
        "held_out_case_count": len(test),
        "action_boundary": "recommendation preference-eligibility epsilon only",
        "vehicle_control": False,
        "mean_ttt_gap_vs_deterministic": mean_gap,
        "p95_inference_seconds": p95_inference,
        "regret_violation_count": regret_violations,
        "safety_violation_count": safety_violations,
        "nonstationarity_superior_to_deterministic": nonstationarity_superior,
        "retained": retained,
        "outcome": "C" if retained else "B",
        "training": telemetry.__dict__,
        "statistics": statistics,
        "claim_boundary": config["claim_boundary"],
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    raw_path = OUTPUT / "raw_metrics.json"
    processed_path = OUTPUT / "processed_metrics.json"
    statistical_path = OUTPUT / "statistical_tests.json"
    summary_path = OUTPUT / "summary.json"
    _write_json(raw_path, raw)
    _write_json(processed_path, summary)
    _write_json(statistical_path, statistics)
    _write_json(summary_path, summary)
    figure_dir = OUTPUT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(6.5, 4.2))
    axis.bar(
        ["Deterministic", "Fixed B6", "RL0"],
        [np.mean(deterministic_ttt), np.mean(b6_ttt), np.mean(rl_ttt)],
        color=["#222222", "#777777", "#bbbbbb"],
    )
    axis.set_ylabel("Held-out analytical TTT")
    axis.set_title("Conditional RL0 comparison")
    fig.tight_layout()
    figure_path = figure_dir / "rl0_comparison.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    ended = datetime.now(timezone.utc)
    outputs = [raw_path, processed_path, statistical_path, summary_path, figure_path]
    run_dir = ExperimentRegistry(str(ROOT / "artifacts" / "runs")).create(
        config,
        summary,
        input_paths=(str(CONFIG.relative_to(ROOT)), config["source_study"]),
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
    run()
