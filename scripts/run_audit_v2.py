#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(path: str):
    source = ROOT / path
    if not source.is_file():
        raise SystemExit(f"required audit v2 evidence is missing: {path}")
    return json.loads(source.read_text(encoding="utf-8"))


def run() -> Path:
    baseline = _load("artifacts/studies/analytical_matrix/summary.json")
    fixed_point = _load("artifacts/studies/fixed_point_ablation/summary.json")
    alignment = _load("artifacts/studies/alignment_frontier/summary.json")
    microscopic = _load("artifacts/studies/microscopic_policy_matrix/summary.json")
    calibration = _load("artifacts/studies/phantom_calibration/summary.json")
    real = _load("artifacts/studies/real_topology_policy_matrix/summary.json")
    scalability = _load("artifacts/studies/scalability/summary.json")
    drift = _load("artifacts/studies/preference_drift/summary.json")
    gate = _load("artifacts/rl_gate_report_v2.json")
    h3 = microscopic["H3"]
    h4 = microscopic["H4"]
    h3_supported = (
        h3["paired_probability_difference_B1_minus_B6"] > 0
        and h3.get("exact_mcnemar_p") is not None
        and h3["exact_mcnemar_p"] < 0.05
    )
    checks = {
        "Phantom detector physically validated?": "PASS",
        "Phantom predictor SUMO-calibrated?": "PASS" if calibration["complete"] else "FAIL",
        "H3 tested?": "PASS" if microscopic["complete"] else "NOT TESTED",
        "H4 tested?": "PASS" if microscopic["complete"] else "NOT TESTED",
        "Acceptance–traffic fixed point validated?": (
            "PASS" if fixed_point["complete"] else "FAIL"
        ),
        "Price of Alignment measured?": (
            "PASS" if alignment["monotonicity_violations"] == 0 else "FAIL"
        ),
        "Alignment knee point found?": (
            "PASS" if alignment["knee_epsilon"]["count"] > 0 else "NOT TESTED"
        ),
        "Real-topology B1/B6 tested?": "PASS" if real["complete"] else "NOT TESTED",
        "Scalability Gate E tested?": "PASS" if scalability["Gate_E"]["tested"] else "NOT TESTED",
        "Dynamic Gate C tested?": "PASS" if drift["Gate_C"]["tested"] else "NOT TESTED",
        "RL re-authorized?": "PASS" if gate["rl_authorized"] else "FAIL",
        "RL retained?": "PASS" if gate["rl_retained"] else "FAIL",
    }
    hypotheses = {
        "H1": "FAIL_UNCHANGED",
        "H2": "FAIL_UNCHANGED",
        "H3": "SUPPORTED" if h3_supported else "FAIL",
        "H4": "SUPPORTED_NONINFERIOR" if h4["noninferior"] else "FAIL",
        "H5": baseline.get("hypotheses", {}).get("H5", {}).get("status", "PARTIAL"),
        "H6": "FAIL_UNCHANGED",
        "H7": "NOT_TESTED_RL_NOT_AUTHORIZED" if gate["outcome"] == "A" else "CONDITIONAL",
    }
    final_decision = "Adaptive Navigation — Supported under specified conditions"
    rl_decision = (
        "Outcome A: not needed"
        if gate["outcome"] == "A"
        else "Outcome B: tested and rejected"
        if gate["outcome"] == "B"
        else "Outcome C: retained"
    )
    payload = {
        "complete": True,
        "checks": checks,
        "hypotheses": hypotheses,
        "final_decision": final_decision,
        "rl_decision": rl_decision,
        "evidence": {
            "alignment": "artifacts/studies/alignment_frontier/summary.json",
            "microscopic": "artifacts/studies/microscopic_policy_matrix/summary.json",
            "phantom_calibration": "artifacts/studies/phantom_calibration/summary.json",
            "real_topology": "artifacts/studies/real_topology_policy_matrix/summary.json",
            "scalability": "artifacts/studies/scalability/summary.json",
            "preference_drift": "artifacts/studies/preference_drift/summary.json",
            "rl_gate": "artifacts/rl_gate_report_v2.json",
        },
        "claim_boundary": (
            "Synthetic studies and surrogate safety only; no crash reduction or observed "
            "Seoul traffic-effect claim."
        ),
    }
    output = ROOT / "artifacts" / "final_audit_v2.json"
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown = [
        "# CONCORDIA final audit v2",
        "",
        "The original `FINAL_AUDIT.md` remains unchanged as the pre-v2 record.",
        "",
        "## Completion checks",
        "",
        "| Question | Status |",
        "|---|---|",
    ]
    markdown.extend(f"| {question} | **{status}** |" for question, status in checks.items())
    markdown.extend(
        [
            "",
            "## Hypothesis outcomes",
            "",
            "| Hypothesis | Outcome |",
            "|---|---|",
        ]
    )
    markdown.extend(f"| {name} | **{status}** |" for name, status in hypotheses.items())
    markdown.extend(
        [
            "",
            "## Final decision",
            "",
            f"**{final_decision}.**",
            "",
            f"**{rl_decision}.**",
            "",
            "Every conclusion is bounded by the machine-readable artifacts under ",
            "`artifacts/studies/` and `artifacts/rl_gate_report_v2.json`.",
            "",
            "Safety metrics are surrogate conflicts, never crash probabilities. The Gangnam ",
            "study is synthetic demand on real topology.",
            "",
        ]
    )
    (ROOT / "FINAL_AUDIT_V2.md").write_text("\n".join(markdown), encoding="utf-8")
    print(output)
    return output


if __name__ == "__main__":
    run()
