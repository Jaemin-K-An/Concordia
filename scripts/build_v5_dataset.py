#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

from concordia.feasibility import (
    BootstrapFeasibilityEnsemble,
    FEATURE_SCHEMA,
    V5_FEATURE_SCHEMA,
    build_alignment_case,
    classify_alignment_case,
    expand_v5_features,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/studies/v5_model_selection"


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _role(seed: int, split: dict, existing: str | None = None) -> str:
    development = split["development"]
    for name in ("training", "calibration_fit", "validation", "shift_validation"):
        if seed in development[f"{name}_seeds"]:
            return name
    if existing == "validation":
        return "validation"
    if existing in {"calibration_evaluation", "calibration_fit"}:
        return "calibration_fit"
    return "training"


def _adjust_case(case: dict, acceptance: float, preference_variance: float) -> dict:
    baseline_ttt = float(case["baseline_metrics"]["eta_only_ttt"])
    original_gain = float(case["adaptive_counterfactual"]["relative_ttt_gain"])
    adjusted_gain = original_gain * acceptance - 0.015 * abs(preference_variance - 1.0)
    adaptive_ttt = baseline_ttt * (1.0 - adjusted_gain)
    original_safety = float(case["adaptive_counterfactual"]["safety_difference"])
    adjusted_safety = original_safety + 0.03 * (1.0 - acceptance)
    accepted = float(case["adaptive_counterfactual"]["acceptance_probability"])
    accepted = float(np.clip(accepted * acceptance, 0.0, 1.0))
    label = classify_alignment_case(
        baseline_ttt=baseline_ttt,
        adaptive_ttt=adaptive_ttt,
        maximum_regret=float(case["adaptive_counterfactual"]["maximum_regret"]),
        regret_limit=0.08,
        baseline_risk=float(case["baseline_metrics"]["safety_risk"]),
        adaptive_risk=float(case["baseline_metrics"]["safety_risk"]) + adjusted_safety,
        safety_delta=0.25,
        legal=True,
        meaningful_intervention=(
            int(case["adaptive_counterfactual"]["beneficial_diversion_count"]) > 0
            and accepted > 0.0
        ),
        minimum_relative_ttt_gain=0.01,
    )
    case["adaptive_counterfactual"].update(
        {
            "ttt": adaptive_ttt,
            "relative_ttt_gain": adjusted_gain,
            "safety_difference": adjusted_safety,
            "acceptance_probability": accepted,
        }
    )
    case["label"] = label.value
    case["condition"].update(
        {
            "acceptance_multiplier": acceptance,
            "preference_variance_multiplier": preference_variance,
        }
    )
    return case


def _upgrade(case: dict, split: dict, source: str) -> dict:
    upgraded = dict(case)
    upgraded["features"] = expand_v5_features(case)
    upgraded["features"]["preference_variance"] *= float(
        case["condition"].get("preference_variance_multiplier", 1.0)
    )
    upgraded["features"]["heterogeneity_rad_interaction"] = (
        upgraded["features"]["preference_variance"]
        * upgraded["features"]["route_attribute_diversity"]
    )
    upgraded["development_role"] = _role(
        int(case["seed"]), split, case.get("development_role")
    )
    upgraded["development_source"] = source
    upgraded["targets"] = {
        "success": int(case["label"] == "WIN"),
        "relative_ttt_gain": float(case["adaptive_counterfactual"]["relative_ttt_gain"]),
        "safety_difference": float(case["adaptive_counterfactual"]["safety_difference"]),
    }
    return upgraded


def run() -> Path:
    if (ROOT / "configs/v5/frozen_model.yaml").is_file():
        existing = OUTPUT / "dataset_summary.json"
        if not existing.is_file():
            raise RuntimeError("v5 is frozen but its dataset is missing")
        print(existing)
        return existing
    split = yaml.safe_load((ROOT / "configs/v5/splits.yaml").read_text())
    prereg = yaml.safe_load((ROOT / "configs/v5/preregistration.yaml").read_text())
    historical = []
    for source in split["historical_sources"]:
        for case in json.loads((ROOT / source).read_text()):
            case = dict(case)
            case["condition"].setdefault("acceptance_multiplier", 1.0)
            case["condition"].setdefault("preference_variance_multiplier", 1.0)
            historical.append(_upgrade(case, split, source))
    v4_selected = json.loads(
        (ROOT / "artifacts/studies/v3_feasibility_prediction/selected_model.json").read_text()
    )
    v4_ensemble = BootstrapFeasibilityEnsemble.from_dict(v4_selected["ensemble"])
    config = split["development"]
    pool = []
    acceptance_values = config["acceptance_multiplier"]
    variance_values = config["preference_variance_multiplier"]
    index = 0
    for scenario in config["scenarios"]:
        for seed in config["seeds"]:
            for demand in config["demand_scale"]:
                for heterogeneity in config["heterogeneity"]:
                    for penetration in config["navigation_penetration"]:
                        acceptance = float(acceptance_values[index % len(acceptance_values)])
                        variance = float(
                            variance_values[(index // len(acceptance_values)) % len(variance_values)]
                        )
                        index += 1
                        case = build_alignment_case(
                            scenario=scenario,
                            seed=int(seed),
                            demand_scale=float(demand),
                            heterogeneity=heterogeneity,
                            navigation_penetration=float(penetration),
                            user_count=6,
                            regret_limit=0.08,
                            epsilon_grid=[0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.24],
                            minimum_relative_ttt_gain=0.01,
                            safety_delta=0.25,
                            source_split="v5_development_pool",
                        )
                        case = _adjust_case(case, acceptance, variance)
                        case["case_id"] += f"-a{acceptance:.2f}-v{variance:.2f}"
                        matrix = np.asarray(
                            [[case["features"][name] for name in FEATURE_SCHEMA]], dtype=float
                        )
                        probability, uncertainty, _lower = v4_ensemble.predict(matrix)
                        relevance = 1.0 - min(1.0, abs(float(probability[0]) - 0.70) / 0.70)
                        failure_region = 1.0 + 0.5 * (
                            penetration == 0.5
                            or scenario == "signalized"
                            or demand >= 1.2
                        )
                        acquisition = float(uncertainty[0]) * relevance * failure_region
                        pool.append((_upgrade(case, split, "v5_active_design"), acquisition))
    by_seed = defaultdict(list)
    for row, acquisition in pool:
        by_seed[int(row["seed"])].append((row, acquisition))
    per_seed = int(config["selected_case_target"]) // len(config["seeds"])
    selected = []
    for values in by_seed.values():
        selected.extend(
            row
            for row, _score in sorted(
                values, key=lambda item: (-item[1], item[0]["case_id"])
            )[:per_seed]
        )
    rows = historical + selected
    final_seeds = set(split["final_analytical_holdout"]["seeds"])
    final_seeds |= set(split["final_stress_holdout"]["seeds"])
    if any(int(row["seed"]) in final_seeds for row in rows):
        raise RuntimeError("v5 final holdout seed entered development data")
    if any(tuple(row["features"]) != V5_FEATURE_SCHEMA for row in rows):
        raise RuntimeError("v5 feature schema order mismatch")
    raw = json.dumps(rows, sort_keys=True, allow_nan=False).encode()
    roles = {
        role: sorted(row["case_id"] for row in rows if row["development_role"] == role)
        for role in sorted({row["development_role"] for row in rows})
    }
    split_hash = _sha_bytes(json.dumps(roles, sort_keys=True).encode())
    schema_hash = _sha_bytes(json.dumps(V5_FEATURE_SCHEMA).encode())
    summary = {
        "complete": True,
        "study": "v5 expanded analytical development dataset",
        "case_count": len(rows),
        "historical_case_count": len(historical),
        "new_selected_case_count": len(selected),
        "new_pool_count": len(pool),
        "role_counts": dict(Counter(row["development_role"] for row in rows)),
        "label_counts": dict(Counter(row["label"] for row in rows)),
        "feature_schema_hash": schema_hash,
        "split_hash": split_hash,
        "raw_hash": _sha_bytes(raw),
        "final_holdouts_materialized": False,
        "rl_used": prereg["rl_used"],
    }
    _write(OUTPUT / "raw_metrics.json", rows)
    _write(OUTPUT / "dataset_summary.json", summary)
    _write(OUTPUT / "split_manifest.json", {"roles": roles, "split_hash": split_hash})
    _write(OUTPUT / "feature_schema.json", {"features": V5_FEATURE_SCHEMA, "hash": schema_hash})
    print(OUTPUT / "dataset_summary.json")
    return OUTPUT / "dataset_summary.json"


if __name__ == "__main__":
    run()
