#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml

from v6_frozen import verify_frozen


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/studies/v7_paired_dataset"


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def run() -> Path:
    manifest = verify_frozen()
    if (ROOT / "artifacts/v7/freeze_manifest.json").exists():
        raise RuntimeError("v7 audit cannot replace an existing frozen study")
    sources = [
        ROOT / "artifacts/studies/v6_micro_dataset/raw_metrics.json",
        ROOT / "artifacts/studies/v6_frozen_micro_holdout/raw_metrics.json",
    ]
    rows = [json.loads(path.read_text()) for path in sources]
    historical_ids = [row["case_id"] for values in rows for row in values]
    if len(historical_ids) != len(set(historical_ids)):
        raise RuntimeError("historical v6 pair IDs are not unique")
    design = yaml.safe_load((ROOT / "configs/v7/paired_design.yaml").read_text())
    final_seeds = set(design["final_holdout_seeds"])
    historical_seeds = {int(row["seed"]) for values in rows for row in values}
    new_seeds = set(design["new_development_seeds"])
    if final_seeds & (historical_seeds | new_seeds):
        raise RuntimeError("v7 final microscopic seed overlaps development")
    summary = {
        "complete": True,
        "study": "v7 entry audit and historical paired-evidence promotion",
        "starting_head": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "v6_freeze_verified": True,
        "v6_freeze_manifest_hash": manifest["manifest_self_hash"],
        "historical_pair_count": len(historical_ids),
        "historical_source_counts": {
            str(path.relative_to(ROOT)): len(values) for path, values in zip(sources, rows)
        },
        "new_development_pair_count_preregistered": design["new_development_pair_count"],
        "total_development_pair_count_preregistered": design["total_development_pair_count"],
        "final_pair_count_preregistered": design["final_holdout_pair_count"],
        "final_seed_overlap_count": 0,
        "v6_final_promoted_only_to_historical_development": True,
        "rl_used": False,
    }
    path = OUTPUT / "entry_audit.json"
    write_json(path, summary)
    print(path)
    return path


if __name__ == "__main__":
    run()
