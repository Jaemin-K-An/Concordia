#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from concordia.feasibility import RegimeDefinition


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "artifacts/studies/v5_model_selection"
OUTPUT = ROOT / "artifacts/studies/v5_shift_detection"


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def run() -> Path:
    if (ROOT / "configs/v5/frozen_regimes.yaml").is_file():
        existing = OUTPUT / "regime_definition.json"
        if not existing.is_file():
            raise RuntimeError("v5 regimes are frozen but discovery evidence is missing")
        print(existing)
        return existing
    config = yaml.safe_load((ROOT / "configs/v5/model.yaml").read_text())
    rows = json.loads((STUDY / "raw_metrics.json").read_text())
    training = [row for row in rows if row["development_role"] == "training"]
    labels = [row["targets"]["success"] for row in training]
    definition = RegimeDefinition.discover(
        training,
        labels,
        cut_candidates=config["regime_cut_candidates"],
        minimum_size=int(config["minimum_regime_size"]),
    )
    enriched = []
    for row in rows:
        value = dict(row)
        value["regime"] = definition.route(row["features"])
        enriched.append(value)
    rates = defaultdict(list)
    for row in training:
        rates[str(row["condition"]["navigation_penetration"])].append(
            row["targets"]["success"]
        )
    response = {
        penetration: {
            "case_count": len(values),
            "success_rate": sum(values) / len(values),
        }
        for penetration, values in sorted(rates.items())
    }
    package = {
        "complete": True,
        "definition": definition.to_dict(),
        "regime_counts": dict(Counter(row["regime"] for row in enriched)),
        "traffic_response_by_penetration": response,
        "discovery_roles": ["training"],
        "final_holdouts_used": False,
    }
    _write(OUTPUT / "regime_definition.json", package)
    _write(STUDY / "regime_rows.json", enriched)
    print(OUTPUT / "regime_definition.json")
    return OUTPUT / "regime_definition.json"


if __name__ == "__main__":
    run()
