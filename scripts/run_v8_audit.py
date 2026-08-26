#!/usr/bin/env python3
from __future__ import annotations

import yaml

from concordia.safety_v8.features import ACTION_AWARE_FEATURE_SCHEMA, FORBIDDEN_INPUT_TOKENS
from v7_frozen import verify_frozen
from v8_common import ROOT, sha256, write_json


def run():
    v7 = verify_frozen()
    design = yaml.safe_load((ROOT / "configs/v8/safety_design.yaml").read_text())
    development = set(map(int, design["new_development_seeds"]))
    final = set(map(int, design["final_holdout_seeds"]))
    v7_final = set(
        map(
            int,
            yaml.safe_load((ROOT / "configs/v7/paired_design.yaml").read_text())["final_holdout_seeds"],
        )
    )
    forbidden_names = [
        name for name in ACTION_AWARE_FEATURE_SCHEMA
        if any(token in name.lower() for token in FORBIDDEN_INPUT_TOKENS)
    ]
    report = {
        "study": "CONCORDIA v8 pre-final integrity audit",
        "v7_manifest_verified": True,
        "v7_manifest_self_hash": v7["manifest_self_hash"],
        "v7_manifest_file_hash": sha256(ROOT / "artifacts/v7/freeze_manifest.json"),
        "v8_preregistration_exists": (ROOT / "configs/v8/preregistration.yaml").is_file(),
        "development_final_seed_overlap": sorted(development & final),
        "v7_v8_final_seed_overlap": sorted(v7_final & final),
        "forbidden_feature_names": forbidden_names,
        "action_aware_feature_count": len(ACTION_AWARE_FEATURE_SCHEMA),
        "final_holdout_exists": (ROOT / "artifacts/studies/v8_micro_holdout/raw_metrics.json").is_file(),
        "final_holdout_exists_without_v8_freeze": bool(
            (ROOT / "artifacts/studies/v8_micro_holdout/raw_metrics.json").is_file()
            and not (ROOT / "artifacts/v8/freeze_manifest.json").is_file()
        ),
        "rl_used": False,
    }
    report["passed"] = bool(
        report["v8_preregistration_exists"]
        and not report["development_final_seed_overlap"]
        and not report["v7_v8_final_seed_overlap"]
        and not forbidden_names
        and not report["final_holdout_exists_without_v8_freeze"]
    )
    write_json(ROOT / "artifacts/studies/v8_audit/summary.json", report)
    if not report["passed"]:
        raise RuntimeError("v8 integrity audit failed")
    print(ROOT / "artifacts/studies/v8_audit/summary.json")


if __name__ == "__main__":
    run()
