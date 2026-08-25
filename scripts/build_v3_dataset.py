#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

from concordia.evaluation import ExperimentRegistry, capture_source_state
from concordia.feasibility import FEATURE_SCHEMA, build_alignment_case


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "studies" / "v3_feasibility_prediction"


def _json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def run() -> Path:
    source_commit, source_dirty = capture_source_state()
    started = datetime.now(timezone.utc)
    prereg = yaml.safe_load((ROOT / "configs/v3/preregistration.yaml").read_text())
    split = yaml.safe_load((ROOT / "configs/v3/splits.yaml").read_text())
    schema = yaml.safe_load((ROOT / "configs/v3/feature_schema.yaml").read_text())
    if tuple(schema["feature_order"]) != FEATURE_SCHEMA:
        raise RuntimeError("feature schema YAML and implementation differ")
    source = json.loads((ROOT / split["development_source"]).read_text(encoding="utf-8"))
    training_seeds = set(split["training"]["seeds"])
    validation_seeds = set(split["validation"]["seeds"])
    holdout_seeds = set(split["holdout"]["seeds"])
    if (training_seeds | validation_seeds) & holdout_seeds:
        raise RuntimeError("holdout seeds overlap development")
    rows = []
    for raw in source:
        seed = int(raw["seed"])
        source_split = "training" if seed in training_seeds else "validation"
        if seed not in training_seeds | validation_seeds:
            raise RuntimeError(f"unregistered v2 seed: {seed}")
        rows.append(
            build_alignment_case(
                scenario=str(raw["scenario"]),
                seed=seed,
                demand_scale=float(raw["demand_scale"]),
                heterogeneity=str(raw["heterogeneity"]),
                navigation_penetration=1.0,
                user_count=6,
                regret_limit=float(prereg["success_definition"]["regret_limit"]),
                epsilon_grid=[point["epsilon"] for point in raw["frontier"]],
                minimum_relative_ttt_gain=float(
                    prereg["primary"]["minimum_relative_ttt_gain"]
                ),
                safety_delta=float(prereg["success_definition"]["safety_delta"]),
                source_split=source_split,
                precomputed=raw,
            )
        )
    split_ids = {
        name: sorted(row["case_id"] for row in rows if row["source_split"] == name)
        for name in ("training", "validation")
    }
    if set(split_ids["training"]) & set(split_ids["validation"]):
        raise RuntimeError("case leakage between training and validation")
    split_payload = {
        "training_ids": split_ids["training"],
        "validation_ids": split_ids["validation"],
        "holdout_spec": split["holdout"],
        "group_unit": split["anti_leakage"]["group_unit"],
    }
    split_bytes = (json.dumps(split_payload, sort_keys=True) + "\n").encode()
    summary = {
        "complete": True,
        "stage": "development_dataset",
        "case_count": len(rows),
        "training_count": len(split_ids["training"]),
        "validation_count": len(split_ids["validation"]),
        "label_counts": dict(Counter(row["label"] for row in rows)),
        "split_hash": _sha_bytes(split_bytes),
        "feature_schema_hash": _sha_bytes(
            (json.dumps(schema, sort_keys=True) + "\n").encode()
        ),
        "holdout_materialized": False,
        "v2_evidence_role": "development_only",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    raw_path = OUTPUT / "raw_metrics.json"
    split_path = OUTPUT / "split_manifest.json"
    schema_path = OUTPUT / "feature_schema.json"
    summary_path = OUTPUT / "dataset_summary.json"
    _json(raw_path, rows)
    _json(split_path, split_payload)
    _json(schema_path, schema)
    _json(summary_path, summary)
    ended = datetime.now(timezone.utc)
    config = {"seeds": sorted(training_seeds | validation_seeds), **split}
    outputs = (raw_path, split_path, schema_path, summary_path)
    run_dir = ExperimentRegistry(str(ROOT / "artifacts/runs")).create(
        config,
        summary,
        input_paths=(
            "configs/v3/preregistration.yaml",
            "configs/v3/splits.yaml",
            "configs/v3/feature_schema.yaml",
            split["development_source"],
        ),
        external_output_paths=tuple(str(path.relative_to(ROOT)) for path in outputs),
        started_at=started,
        ended_at=ended,
        source_commit=source_commit,
        source_dirty=source_dirty,
    )
    shutil.copyfile(run_dir / "manifest.json", OUTPUT / "dataset_manifest.json")
    print(summary_path)
    return summary_path


if __name__ == "__main__":
    run()
