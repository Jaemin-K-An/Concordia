#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "FINAL_AUDIT_V2.md",
    "artifacts/studies/alignment_frontier/raw_metrics.json",
    "artifacts/studies/microscopic_policy_matrix/summary.json",
    "artifacts/studies/phantom_calibration/summary.json",
    "artifacts/studies/real_topology_policy_matrix/summary.json",
    "artifacts/studies/scalability/summary.json",
    "artifacts/studies/preference_drift/summary.json",
    "artifacts/studies/conditional_rl/summary.json",
    "artifacts/rl_gate_report_v2.json",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run() -> Path:
    missing = [path for path in REQUIRED if not (ROOT / path).is_file()]
    if missing:
        raise SystemExit(f"missing v2 evidence: {missing}")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    audit = (ROOT / "FINAL_AUDIT_V2.md").read_text(encoding="utf-8")
    preserved = all(
        token in audit
        for token in (
            "H1 | **FAIL_UNCHANGED**",
            "H2 | **FAIL_UNCHANGED**",
            "H3 | **FAIL**",
            "H4 | **FAIL**",
            "H6 | **FAIL_UNCHANGED**",
        )
    )
    payload = {
        "complete": True,
        "head": head,
        "v2_failures_preserved": preserved,
        "v2_role": "pilot_development_evidence",
        "evidence_hashes": {path: _sha(ROOT / path) for path in REQUIRED},
        "v3_question": "Know when optimization is worth attempting.",
        "holdout_executed": False,
    }
    output = ROOT / "artifacts" / "v3" / "current_head_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    document = ROOT / "docs" / "v3_preregistered_plan.md"
    document.write_text(
        "# CONCORDIA v3 preregistered plan\n\n"
        f"- Starting HEAD: `{head}`\n"
        "- v2 status: pilot/development evidence; H1/H2/H3/H4/H6 failures preserved.\n"
        "- Primary success: ≥1% B1 TTT gain, regret ≤0.08, safety difference ≤0.25, "
        "legal and accepted-only execution.\n"
        "- Engineering target: intervention precision ≥0.65 and coverage ≥0.40.\n"
        "- Scientific target: lower 95% precision CI >0.50.\n"
        "- Freeze order: development → validation → model/threshold freeze → holdout.\n"
        "- Abstention is neither success nor failure and remains in coverage/PBR denominators.\n"
        "- RL0 remains rejected; v3 ML is limited to feasibility prediction.\n",
        encoding="utf-8",
    )
    print(output)
    return output


if __name__ == "__main__":
    run()
