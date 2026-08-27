#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean, pstdev
from typing import Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
STUDY_DIR = ROOT / "artifacts/studies/v10_racing_validation"
NULL_ACTION = "A00_NULL_B1"


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _safe_success(outcome: Mapping[str, object]) -> bool:
    return bool(
        float(outcome["traffic_gain"]) > 0.005
        and float(outcome["risk_delta"]) <= 0.25
        and float(outcome["maximum_regret"]) <= 0.08
        and bool(outcome["legal"])
    )


def _unsafe(outcome: Mapping[str, object]) -> bool:
    return bool(
        float(outcome["risk_delta"]) > 0.25
        or float(outcome["maximum_regret"]) > 0.08
        or not bool(outcome["legal"])
    )


def _group_stage(record: Mapping[str, object], stage: str) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for result in record["trace"]["rollout_results"]:
        if result["stage"] == stage:
            grouped[result["action_id"]].append(result)
    return grouped


def _stage1_selection(record: Mapping[str, object]) -> str:
    candidates = []
    for action_id, values in _group_stage(record, "stage_1").items():
        result = values[0]
        if _unsafe(result):
            continue
        score = (
            float(result["traffic_gain"])
            - 0.10 * max(0.0, float(result["risk_delta"]))
            - 0.01 * max(0.0, float(result["queue_delta"]))
            - 0.01 * max(0.0, float(result["bottleneck_load_delta"]))
        )
        if float(result["traffic_gain"]) > 0.005:
            candidates.append((score, action_id))
    return max(candidates, default=(-1e9, NULL_ACTION))[1]


def _stage12_selection(record: Mapping[str, object]) -> str:
    candidates = []
    for action_id, values in _group_stage(record, "stage_2").items():
        if sum(_unsafe(value) for value in values) >= 2:
            continue
        traffic = [float(value["traffic_gain"]) for value in values]
        score = fmean(traffic) - 0.50 * pstdev(traffic)
        if score > 0.005:
            candidates.append((score, action_id))
    return max(candidates, default=(-1e9, NULL_ACTION))[1]


def _single_rollout_selection(record: Mapping[str, object]) -> str:
    candidates = []
    for action_id, values in _group_stage(record, "stage_3").items():
        result = sorted(values, key=lambda value: value["replica"])[0]
        if not _unsafe(result) and float(result["traffic_gain"]) > 0.005:
            candidates.append((float(result["traffic_gain"]), action_id))
    return max(candidates, default=(-1e9, NULL_ACTION))[1]


def _no_safety_selection(record: Mapping[str, object]) -> str:
    candidates = [
        (float(statistics["q10_lcb"]), action_id)
        for action_id, statistics in record["trace"]["stage_3_statistics"].items()
        if float(statistics["q10_lcb"]) > 0.005
    ]
    return max(candidates, default=(-1e9, NULL_ACTION))[1]


def _no_lcb_selection(record: Mapping[str, object]) -> str:
    candidates = []
    for action_id, statistics in record["trace"]["stage_3_statistics"].items():
        if (
            int(statistics["unsafe_count"]) == 0
            and float(statistics["maximum_risk_delta"]) <= 0.25
            and float(statistics["maximum_regret"]) <= 0.08
            and bool(statistics["all_legal"])
            and float(statistics["mean_traffic_gain"]) > 0.005
        ):
            candidates.append((float(statistics["mean_traffic_gain"]), action_id))
    return max(candidates, default=(-1e9, NULL_ACTION))[1]


def _policy_metrics(
    records: Sequence[Mapping[str, object]],
    selector: Callable[[Mapping[str, object]], str],
) -> dict:
    selections = [selector(record) for record in records]
    interventions = [
        (record, action_id)
        for record, action_id in zip(records, selections)
        if action_id != NULL_ACTION
    ]
    realized = [record["action_outcomes"][action_id] for record, action_id in interventions]
    successes = sum(_safe_success(outcome) for outcome in realized)
    safety_violation_count = sum(
        float(outcome["risk_delta"]) > 0.25 for outcome in realized
    )
    constraint_violation_count = sum(_unsafe(outcome) for outcome in realized)
    selected_gains = [
        float(record["action_outcomes"][action_id]["traffic_gain"])
        if action_id != NULL_ACTION else 0.0
        for record, action_id in zip(records, selections)
    ]
    action_counts = Counter(action_id for action_id in selections if action_id != NULL_ACTION)
    return {
        "state_count": len(records),
        "intervention_count": len(interventions),
        "coverage": len(interventions) / len(records) if records else 0.0,
        "safe_beneficial_intervention_count": successes,
        "precision": successes / len(interventions) if interventions else 0.0,
        "realized_safety_violation_count": safety_violation_count,
        "realized_any_constraint_violation_count": constraint_violation_count,
        "population_mean_relative_ttt_gain": fmean(selected_gains) if selected_gains else 0.0,
        "selected_action_diversity": len(action_counts),
        "selected_action_counts": dict(sorted(action_counts.items())),
    }


def _survival(records: Sequence[Mapping[str, object]]) -> dict:
    actionable = [record for record in records if record["oracle_beneficial"]]
    stages = ("stage_1", "stage_2", "stage_3", "verification")
    return {
        "oracle_actionable_state_count": len(actionable),
        "rates": {
            stage: (
                sum(bool(record["oracle_survival"][stage]) for record in actionable)
                / len(actionable)
                if actionable else 0.0
            )
            for stage in stages
        },
        "targets": {
            "stage_1": 0.95,
            "stage_2": 0.90,
            "stage_3": 0.80,
            "verification": None,
        },
    }


def _write_survival_svg(path: Path, survival: Mapping[str, object]) -> None:
    names = ["Stage 1", "Stage 2", "Stage 3", "Verify"]
    values = list(survival["rates"].values())
    points = []
    for index, value in enumerate(values):
        x = 90 + index * 145
        y = 260 - 210 * float(value)
        points.append((x, y, float(value)))
    polyline = " ".join(f"{x},{y:.1f}" for x, y, _value in points)
    marks = "".join(
        f'<circle cx="{x}" cy="{y:.1f}" r="5" fill="#2dd4bf"/>'
        f'<text x="{x}" y="{y - 12:.1f}" text-anchor="middle" fill="#e2e8f0" '
        f'font-size="14">{value:.1%}</text>'
        f'<text x="{x}" y="286" text-anchor="middle" fill="#94a3b8" '
        f'font-size="13">{name}</text>'
        for (x, y, value), name in zip(points, names)
    )
    path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="620" height="320" '
        'viewBox="0 0 620 320">'
        '<rect width="620" height="320" rx="16" fill="#0f172a"/>'
        '<text x="28" y="34" fill="#f8fafc" font-size="18" '
        'font-family="sans-serif">V10 Oracle Action Survival Curve</text>'
        '<line x1="55" y1="260" x2="570" y2="260" stroke="#475569"/>'
        '<line x1="55" y1="50" x2="55" y2="260" stroke="#475569"/>'
        f'<polyline points="{polyline}" fill="none" stroke="#2dd4bf" '
        'stroke-width="3"/>'
        f'{marks}</svg>\n',
        encoding="utf-8",
    )


def run(records_path: Path, output_prefix: str) -> None:
    records = json.loads(records_path.read_text())

    def full(record: Mapping[str, object]) -> str:
        return str(record["trace"]["selected_action_id"])

    def b6(_record: Mapping[str, object]) -> str:
        return "B6_ALWAYS_ON_REFERENCE"
    baselines = {
        "B1": _policy_metrics(records, lambda _record: NULL_ACTION),
        "B6": _policy_metrics(records, b6),
        "V10-Stage1": _policy_metrics(records, _stage1_selection),
        "V10-Stage12": _policy_metrics(records, _stage12_selection),
        "V10-F": _policy_metrics(records, full),
        "V9-Surrogate": {
            "comparison_type": "historical_independent_validation",
            "top_5_oracle_recall": 0.375,
            "source": "artifacts/studies/v9_policy_validation/gate_validation.json",
            "contemporaneous_realized_precision": None,
        },
    }
    ablations = {
        "no_replication": _policy_metrics(records, _single_rollout_selection),
        "no_safety_check": _policy_metrics(records, _no_safety_selection),
        "no_robust_lcb": _policy_metrics(records, _no_lcb_selection),
        "full_v10": baselines["V10-F"],
    }
    survival = _survival(records)
    failure_counts = Counter(str(record["failure_taxonomy"]) for record in records)
    oracle_capture = sum(
        record["oracle_beneficial"]
        and record["trace"]["selected_action_id"] == record["oracle_action_id"]
        for record in records
    )
    rollout_count = sum(len(record["trace"]["rollout_results"]) for record in records)
    diagnostics = {
        "failure_taxonomy": dict(sorted(failure_counts.items())),
        "mean_selection_regret": fmean(float(record["selection_regret"]) for record in records),
        "exact_oracle_capture_count": oracle_capture,
        "rollout_count": rollout_count,
        "racing_efficiency_exact_capture_per_rollout": oracle_capture / rollout_count,
        "final_holdout_materialized": False,
    }
    _write(STUDY_DIR / f"{output_prefix}_baseline_comparison.json", baselines)
    _write(STUDY_DIR / f"{output_prefix}_ablations.json", ablations)
    _write(STUDY_DIR / f"{output_prefix}_action_survival.json", survival)
    _write(STUDY_DIR / f"{output_prefix}_diagnostics.json", diagnostics)
    _write_survival_svg(STUDY_DIR / f"{output_prefix}_action_survival.svg", survival)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", type=Path)
    parser.add_argument("--output-prefix", default="development")
    arguments = parser.parse_args()
    run(arguments.records, arguments.output_prefix)


if __name__ == "__main__":
    main()
