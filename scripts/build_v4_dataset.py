#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from concordia.evaluation import ExperimentRegistry, capture_source_state
from concordia.feasibility import (
    BootstrapFeasibilityEnsemble,
    FEATURE_SCHEMA,
    V4_FEATURE_SCHEMA,
    build_alignment_case,
    expand_v4_features,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/studies/v4_model_selection"


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _role(seed: int, split: dict) -> str:
    for role, seeds in split["roles"].items():
        if seed in seeds:
            return role.removesuffix("_seeds")
    raise RuntimeError(f"development seed {seed} has no registered role")


def _upgrade(case: dict, source: str, role: str, uncertainty: float | None = None) -> dict:
    old_features = {name: float(case["features"][name]) for name in FEATURE_SCHEMA}
    upgraded = dict(case)
    upgraded["features_v3"] = old_features
    upgraded["features"] = expand_v4_features(case)
    upgraded["development_source"] = source
    upgraded["development_role"] = role
    upgraded["active_design_uncertainty"] = uncertainty
    upgraded["targets"] = {
        "success": int(case["label"] == "WIN"),
        "relative_ttt_gain": float(case["adaptive_counterfactual"]["relative_ttt_gain"]),
        "absolute_ttt_gain": float(case["baseline_metrics"]["eta_only_ttt"])
        - float(case["adaptive_counterfactual"]["ttt"]),
        "safety_difference": float(case["adaptive_counterfactual"]["safety_difference"]),
        "safety_violation": int(case["adaptive_counterfactual"]["safety_difference"] > 0.25),
    }
    return upgraded


def run() -> Path:
    source_commit, source_dirty = capture_source_state()
    started = datetime.now(timezone.utc)
    split = yaml.safe_load((ROOT / "configs/v4/splits.yaml").read_text(encoding="utf-8"))
    prereg = yaml.safe_load(
        (ROOT / "configs/v4/preregistration.yaml").read_text(encoding="utf-8")
    )
    holdout_seeds = set(split["final_holdout"]["seeds"])
    fitting_seeds = {
        seed for values in split["roles"].values() for seed in values
    }
    if holdout_seeds & fitting_seeds:
        raise RuntimeError("v4 final holdout seed overlaps a development role")
    historical = []
    for source_path in split["historical_sources"]:
        source_rows = json.loads((ROOT / source_path).read_text(encoding="utf-8"))
        for case in source_rows:
            historical.append(
                _upgrade(case, source_path, _role(int(case["seed"]), split))
            )
    v3_selected = json.loads(
        (
            ROOT / "artifacts/studies/v3_feasibility_prediction/selected_model.json"
        ).read_text(encoding="utf-8")
    )
    v3_ensemble = BootstrapFeasibilityEnsemble.from_dict(v3_selected["ensemble"])
    active_config = split["active_design"]
    active_pool = []
    for scenario in active_config["scenarios"]:
        for seed in active_config["seeds"]:
            for demand_scale in active_config["demand_scale"]:
                for heterogeneity in active_config["heterogeneity"]:
                    for penetration in active_config["navigation_penetration"]:
                        case = build_alignment_case(
                            scenario=str(scenario),
                            seed=int(seed),
                            demand_scale=float(demand_scale),
                            heterogeneity=str(heterogeneity),
                            navigation_penetration=float(penetration),
                            user_count=6,
                            regret_limit=float(prereg["success_definition"]["regret_limit"]),
                            epsilon_grid=[0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.24],
                            minimum_relative_ttt_gain=float(
                                prereg["success_definition"]["minimum_relative_ttt_gain"]
                            ),
                            safety_delta=float(prereg["success_definition"]["safety_delta"]),
                            source_split="v4_active_pool",
                        )
                        matrix = np.asarray(
                            [[case["features"][name] for name in FEATURE_SCHEMA]], dtype=float
                        )
                        _mean, uncertainty, _lower = v3_ensemble.predict(matrix)
                        active_pool.append((case, float(uncertainty[0])))
    by_group = defaultdict(list)
    for case, uncertainty in active_pool:
        by_group[(case["seed"], case["scenario"])].append((case, uncertainty))
    selected_active = []
    excluded_active = []
    for values in by_group.values():
        ordered = sorted(values, key=lambda item: (-item[1], item[0]["case_id"]))
        selected_active.extend(ordered[:24])
        excluded_active.extend(ordered[24:])
    active = [
        _upgrade(case, "active_uncertainty_design", _role(int(case["seed"]), split), uncertainty)
        for case, uncertainty in selected_active
    ]
    rows = historical + active
    if any(int(row["seed"]) in holdout_seeds for row in rows):
        raise RuntimeError("final holdout seed entered v4 development data")
    if any(tuple(row["features"]) != V4_FEATURE_SCHEMA for row in rows):
        raise RuntimeError("v4 feature order does not match the preregistered schema")
    split_payload = {
        "roles": {
            role: sorted(row["case_id"] for row in rows if row["development_role"] == role)
            for role in sorted({row["development_role"] for row in rows})
        },
        "final_holdout_spec": split["final_holdout"],
        "final_holdout_case_ids": [],
        "group_unit": split["anti_leakage"]["group_unit"],
    }
    split_bytes = (json.dumps(split_payload, sort_keys=True) + "\n").encode()
    schema_payload = {"feature_order": list(V4_FEATURE_SCHEMA)}
    schema_bytes = (json.dumps(schema_payload, sort_keys=True) + "\n").encode()
    summary = {
        "complete": True,
        "study": "Study X — Robust Feasibility Model Selection dataset",
        "case_count": len(rows),
        "role_counts": dict(Counter(row["development_role"] for row in rows)),
        "source_counts": dict(Counter(row["development_source"] for row in rows)),
        "label_counts": dict(Counter(row["label"] for row in rows)),
        "active_pool_count": len(active_pool),
        "active_selected_count": len(active),
        "active_selection_rule": "top 24 v3 ensemble uncertainty cases within each seed-scenario group; labels unused",
        "final_holdout_materialized": False,
        "split_hash": _sha_bytes(split_bytes),
        "feature_schema_hash": _sha_bytes(schema_bytes),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    raw_path = OUTPUT / "raw_metrics.json"
    split_path = OUTPUT / "split_manifest.json"
    schema_path = OUTPUT / "feature_schema.json"
    active_path = OUTPUT / "active_design.json"
    summary_path = OUTPUT / "dataset_summary.json"
    _write(raw_path, rows)
    _write(split_path, split_payload)
    _write(schema_path, schema_payload)
    _write(
        active_path,
        {
            "pool_count": len(active_pool),
            "selected": [
                {"case_id": case["case_id"], "uncertainty": uncertainty}
                for case, uncertainty in selected_active
            ],
            "excluded": [
                {"case_id": case["case_id"], "uncertainty": uncertainty}
                for case, uncertainty in excluded_active
            ],
            "target_blind_selection": True,
        },
    )
    _write(summary_path, summary)
    ended = datetime.now(timezone.utc)
    outputs = (raw_path, split_path, schema_path, active_path, summary_path)
    registry = ExperimentRegistry(str(ROOT / "artifacts/runs")).create(
        {"seeds": sorted(fitting_seeds), **split},
        summary,
        input_paths=(
            "configs/v4/preregistration.yaml",
            "configs/v4/splits.yaml",
            "configs/v4/feature_schema.yaml",
            *split["historical_sources"],
        ),
        external_output_paths=tuple(str(path.relative_to(ROOT)) for path in outputs),
        started_at=started,
        ended_at=ended,
        source_commit=source_commit,
        source_dirty=source_dirty,
    )
    shutil.copyfile(registry / "manifest.json", OUTPUT / "dataset_manifest.json")
    print(summary_path)
    return summary_path


if __name__ == "__main__":
    run()
