#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/v5/current_head_audit.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run() -> Path:
    required = [
        "FINAL_AUDIT_V3.md",
        "FINAL_AUDIT_V4.md",
        "artifacts/studies/v4_frozen_holdout/summary.json",
        "artifacts/studies/v4_microscopic/summary.json",
        "artifacts/studies/v4_real_topology/summary.json",
        "artifacts/studies/v4_stress/summary.json",
        "configs/v4/frozen_model.yaml",
        "configs/v4/frozen_thresholds.yaml",
    ]
    missing = [path for path in required if not (ROOT / path).is_file()]
    if missing:
        raise RuntimeError(f"required v4 evidence is missing: {missing}")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    holdout = json.loads(
        (ROOT / "artifacts/studies/v4_frozen_holdout/summary.json").read_text()
    )
    micro = json.loads(
        (ROOT / "artifacts/studies/v4_microscopic/summary.json").read_text()
    )
    recomputed_claim = (
        micro["policy_metrics"]["V4-F"]["intervention_count"] > 0
        and micro["policy_metrics"]["V4-F"]["successful_intervention_count"] > 0
        and micro["policy_metrics"]["V4-F"]["safety_violation_count"] == 0
    )
    payload = {
        "complete": True,
        "starting_head": head,
        "v4_outcome_preserved": holdout["outcome"] == "P",
        "v4_precision": holdout["primary_metrics"]["intervention_precision"],
        "v4_coverage": holdout["primary_metrics"]["coverage"],
        "v4_claim_flag_stored": micro["adaptive_success_claim_allowed"],
        "v4_claim_flag_recomputed": recomputed_claim,
        "v4_claim_flag_bug_confirmed": micro["adaptive_success_claim_allowed"]
        != recomputed_claim,
        "v4_role": "pilot and historical development evidence only",
        "v5_final_holdouts_materialized": False,
        "rl_used": False,
        "evidence_hashes": {path: _sha(ROOT / path) for path in required},
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    plan = ROOT / "docs/v5_preregistered_plan.md"
    plan.write_text(
        "# CONCORDIA v5 preregistered plan\n\n"
        f"- Starting HEAD: `{head}`\n"
        "- v2–v4 outcomes and artifacts remain unchanged.\n"
        "- v4 is pilot/historical development evidence; two entirely new v5 holdouts follow freeze.\n"
        "- Analytical target: precision ≥0.80, coverage ≥0.15 (strong ≥0.20), N intervention ≥75.\n"
        "- Stress target: precision ≥0.70, coverage >0, analytical safety violations 0.\n"
        "- Microscopic target: ≥10 interventions, precision >0.50, safe successes >0, safety violations 0.\n"
        "- Regimes, DSS, calibration, correction, veto and every threshold freeze before either holdout.\n"
        "- RL remains excluded.\n"
    )
    print(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    run()
