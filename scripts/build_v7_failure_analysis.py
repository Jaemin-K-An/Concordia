#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from v7_frozen import verify_frozen, write_json


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/studies/v7_frozen_micro_holdout"
OUTPUT = ROOT / "artifacts/studies/v7_failure_analysis"


def _false_positive_taxonomy(row: dict) -> str:
    outcome = row["outcomes"]
    condition = row["condition"]
    if float(outcome["tau_s"]) > 0.25:
        return "safety-effect underestimation"
    if float(outcome["max_regret"]) > 0.08:
        return "regret underestimation"
    if float(outcome["tau_t_relative"]) < 0.01:
        if float(condition["acceptance_multiplier"]) < 0.8:
            return "partial adoption mismatch"
        if condition["topology"] in {"asymmetric", "real_like"}:
            return "topology transfer failure"
        if condition["perturbation"] in {"medium", "strong"}:
            return "secondary bottleneck"
        return "uplift overestimation"
    return "temporal nonstationarity"


def _false_negative_taxonomy(row: dict, decision: dict) -> str:
    if row["condition"]["topology"] in {"asymmetric", "real_like"}:
        return "topology extrapolation"
    if decision["reason"] == "safety_effect_veto":
        return "safety uncertainty"
    if float(row["condition"]["penetration"]) in {0.25, 1.0}:
        return "penetration regime"
    if decision["reason"] == "traffic_uplift_uncertain":
        if float(row["outcomes"]["tau_t_relative"]) >= 0.03:
            return "conservative quantile bound"
        return "weak uplift uncertainty"
    return "calibration failure"


def _group_effects(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        demand = float(row["condition"]["demand"])
        demand_band = "low" if demand < 1000 else "medium" if demand <= 1300 else "high"
        keys = (
            ("topology", str(row["condition"]["topology"])),
            ("penetration", str(row["condition"]["penetration"])),
            ("demand_band", demand_band),
            ("acceptance", str(row["condition"]["acceptance_multiplier"])),
        )
        for axis, value in keys:
            groups[(axis, value)].append(float(row["outcomes"]["tau_t_relative"]))
    return [
        {
            "axis": axis,
            "value": value,
            "count": len(values),
            "mean_relative_treatment_effect": float(np.mean(values)),
            "median_relative_treatment_effect": float(np.median(values)),
            "negative_effect_rate": float(np.mean(np.asarray(values) < 0.0)),
            "structurally_negative_descriptive": bool(
                len(values) >= 10 and float(np.mean(values)) < 0.0
            ),
        }
        for (axis, value), values in sorted(groups.items())
    ]


def run() -> Path:
    existing = OUTPUT / "summary.json"
    if existing.is_file():
        verify_frozen()
        print(existing)
        return existing
    verify_frozen()
    rows = json.loads((SOURCE / "raw_metrics.json").read_text())
    decisions = json.loads((SOURCE / "decision_log.json").read_text())
    decision_by_id = {row["pair_id"]: row for row in decisions}
    false_positive = []
    false_negative = []
    for row in rows:
        decision = decision_by_id[row["pair_id"]]
        selected = bool(decision["intervene"])
        success = bool(row["outcomes"]["safe_micro_success"])
        if selected and not success:
            false_positive.append(
                {
                    "pair_id": row["pair_id"],
                    "taxonomy": _false_positive_taxonomy(row),
                }
            )
        if not selected and success:
            false_negative.append(
                {
                    "pair_id": row["pair_id"],
                    "taxonomy": _false_negative_taxonomy(row, decision),
                }
            )
    group_effects = _group_effects(rows)
    summary = {
        "complete": True,
        "false_positive_count": len(false_positive),
        "false_negative_count": len(false_negative),
        "false_positive_taxonomy": dict(
            Counter(row["taxonomy"] for row in false_positive)
        ),
        "missed_opportunity_taxonomy": dict(
            Counter(row["taxonomy"] for row in false_negative)
        ),
        "structurally_negative_condition_count": sum(
            row["structurally_negative_descriptive"] for row in group_effects
        ),
        "claim_boundary": "negative regimes are descriptive within randomized synthetic SUMO conditions",
    }
    write_json(OUTPUT / "false_positive_cases.json", false_positive)
    write_json(OUTPUT / "missed_opportunity_cases.json", false_negative)
    write_json(OUTPUT / "treatment_effect_by_condition.json", group_effects)
    write_json(OUTPUT / "summary.json", summary)
    print(OUTPUT / "summary.json")
    return OUTPUT / "summary.json"


if __name__ == "__main__":
    run()
