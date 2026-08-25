#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path

from v5_frozen import verify_frozen


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/studies/v6_micro_dataset"


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def run() -> Path:
    manifest = verify_frozen()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    rows = json.loads(
        (ROOT / "artifacts/studies/v5_microscopic_holdout/raw_metrics.json").read_text()
    )
    hard_cases = []
    for row in rows:
        if not row["intervene"] and row["counterfactual_success"]:
            kind = "hard_positive"
        elif row["intervene"] and not row["counterfactual_success"]:
            kind = "hard_negative"
        elif row["intervene"] and row["counterfactual_success"]:
            kind = "selected_positive"
        else:
            kind = "ordinary_negative"
        hard_cases.append(
            {
                "case_id": row["case_id"],
                "kind": kind,
                "scenario": row["scenario"],
                "seed": row["seed"],
                "demand_vehicles_per_hour": row["demand_vehicles_per_hour"],
                "navigation_penetration": row["navigation_penetration"],
                "heterogeneity": row["heterogeneity"],
                "predicted_probability": row["microscopic_success_probability"],
                "traffic_gain": row["microscopic_benefit"],
                "safety_violation": row["microscopic_safety_difference"] > 0.25,
                "safe_micro_success": bool(row["counterfactual_success"]),
            }
        )
    counts = Counter(row["kind"] for row in hard_cases)
    summary = {
        "complete": True,
        "study": "v6 entry audit and v5 hard-case extraction",
        "starting_head": head,
        "v5_freeze_verified": True,
        "v5_freeze_source_commit": manifest["source_commit"],
        "v5_historical_case_count": len(hard_cases),
        "hard_case_counts": dict(counts),
        "hard_positive_by_scenario": dict(
            Counter(row["scenario"] for row in hard_cases if row["kind"] == "hard_positive")
        ),
        "hard_negative_by_scenario": dict(
            Counter(row["scenario"] for row in hard_cases if row["kind"] == "hard_negative")
        ),
        "historical_cases_used_for_final_holdout": False,
        "rl_used": False,
    }
    _write(OUTPUT / "v5_hard_cases.json", hard_cases)
    _write(OUTPUT / "v5_hard_case_summary.json", summary)
    plan = ROOT / "docs/v6_preregistered_plan.md"
    plan.write_text(
        """# CONCORDIA v6 Preregistered Plan

v6 replaces analytical-to-microscopic correction with direct prediction of SafeMicroSuccess.
The new actual-SUMO development set has 600 paired conditions split by seed family into
360 train, 120 calibration, and 120 validation pairs. A new 200-pair final micro holdout uses
ten disjoint seeds. Features are restricted to state observed no later than the 30-second
decision timestamp. Model, calibration, feature subset, safety architecture, conformal cutoff,
and threshold are selected before five YAML packages and a SHA-256 manifest are frozen.

Primary targets are micro precision 0.80, coverage 0.10, at least 30 interventions,
Opportunity Recovery Rate 0.40, zero safety violations, and false-safe rate at most 0.05.
The outcome is reported unchanged if these targets fail. RL remains excluded.
""",
        encoding="utf-8",
    )
    print(OUTPUT / "v5_hard_case_summary.json")
    return OUTPUT / "v5_hard_case_summary.json"


if __name__ == "__main__":
    run()
