#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "FINAL_AUDIT_V2.md",
    "FINAL_AUDIT_V3.md",
    "artifacts/studies/v3_feasibility_prediction/summary.json",
    "artifacts/studies/v3_selective_holdout/summary.json",
    "artifacts/studies/v3_microscopic_selective/summary.json",
    "artifacts/studies/v3_real_topology_selective/summary.json",
    "artifacts/studies/v3_tail_robustness/summary.json",
    "configs/v3/frozen_thresholds.yaml",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run() -> Path:
    missing = [path for path in REQUIRED if not (ROOT / path).is_file()]
    if missing:
        raise RuntimeError(f"required v2/v3 evidence is missing: {missing}")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    v3 = json.loads(
        (ROOT / "artifacts/studies/v3_selective_holdout/summary.json").read_text(
            encoding="utf-8"
        )
    )
    payload = {
        "complete": True,
        "starting_head": head,
        "v3_outcome_preserved": v3["outcome"] == "P",
        "v3_precision": v3["primary_metrics"]["intervention_precision"],
        "v3_coverage": v3["primary_metrics"]["coverage"],
        "v3_holdout_role_v4": "historical development only",
        "v4_final_holdout_materialized": False,
        "rl_used": False,
        "evidence_hashes": {path: _sha(ROOT / path) for path in REQUIRED},
    }
    output = ROOT / "artifacts/v4/current_head_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plan = ROOT / "docs/v4_preregistered_plan.md"
    plan.write_text(
        "# CONCORDIA v4 preregistered plan\n\n"
        f"- Starting HEAD: `{head}`\n"
        "- v2/v3 outcomes remain unchanged; v3 holdout is historical development only.\n"
        "- Primary constraint: holdout precision ≥0.80 while maximizing coverage.\n"
        "- Coverage minimum/stretched targets: 0.20/0.25; guard during selection: 0.15.\n"
        "- Desired/minimum intervention counts: 50/30.\n"
        "- Strong/very strong lower 95% precision bounds: >0.60/>0.70.\n"
        "- Per-intervention success: relative TTT gain ≥0.01, regret ≤0.08, safety Δ≤0.25, legal/executable/accepted-only.\n"
        "- Freeze order: development → robust CV → calibration → validation → commit freeze → new holdout.\n"
        "- Abstentions are excluded from intervention precision and retained in coverage/PBR denominators.\n"
        "- RL remains rejected and absent.\n",
        encoding="utf-8",
    )
    print(output)
    return output


if __name__ == "__main__":
    run()
