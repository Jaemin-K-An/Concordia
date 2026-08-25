#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(path: str):
    source = ROOT / path
    if not source.is_file():
        raise SystemExit(f"required RL Gate v2 evidence is missing: {path}")
    return json.loads(source.read_text(encoding="utf-8"))


def run() -> Path:
    original = _load("artifacts/rl_gate_report.json")
    microscopic = _load("artifacts/studies/microscopic_policy_matrix/summary.json")
    real = _load("artifacts/studies/real_topology_policy_matrix/summary.json")
    scalability = _load("artifacts/studies/scalability/summary.json")
    drift = _load("artifacts/studies/preference_drift/summary.json")
    fixed_point_tested = (ROOT / "tests" / "test_validation_v2.py").is_file()
    gates = {
        "A_performance_gap": original["gates"]["A_performance_gap"],
        "B_runtime_small_scale": original["gates"]["B_runtime"],
        "C_dynamic_generalization": {
            **drift["Gate_C"],
            "triggered": drift["Gate_C"]["triggered_for_RL"],
        },
        "D_feedback_instability": original["gates"]["D_feedback_instability"],
        "E_scalability": {
            **scalability["Gate_E"],
            "triggered": scalability["Gate_E"]["triggered_for_RL"],
            "interpretation": (
                "B6 enumeration failed at scale, but RL is triggered only if a comparable "
                "hard-constrained mathematical approximation leaves a residual latency/quality "
                "problem."
            ),
        },
    }
    triggered = [name for name, gate in gates.items() if gate.get("triggered", False)]
    authorized = bool(triggered)
    result = {
        "outcome": "AUTHORIZED_PENDING_EVALUATION" if authorized else "A",
        "decision": (
            "RL is authorized and requires the conditional baseline evaluation."
            if authorized
            else "Outcome A: RL not needed after measured Gate C/E and the scalable "
            "mathematical approximation."
        ),
        "rl_authorized": authorized,
        "rl_introduced": False,
        "rl_retained": False,
        "triggered_gates": triggered,
        "gates": gates,
        "prerequisites": {
            "microscopic_H3_H4_complete": microscopic.get("complete", False),
            "acceptance_traffic_fixed_point_validated": fixed_point_tested,
            "real_topology_complete": real.get("complete", False),
            "scalability_complete": scalability.get("complete", False),
            "nonstationarity_complete": drift.get("complete", False),
        },
        "thresholds_frozen_before_v2_decision": {
            "dynamic_degradation_relative": 0.10,
            "operational_latency_seconds": 5.0,
            "approximation_quality_gap_relative": 0.05,
        },
        "claim_boundary": (
            "Outcome A means RL was unnecessary for the tested synthetic analytical and "
            "microscopic conditions; it is not a universal impossibility claim."
        ),
    }
    output = ROOT / "artifacts" / "rl_gate_report_v2.json"
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    document = ROOT / "docs" / "rl_gate_decision_v2.md"
    rows = [
        "# RL gate decision v2",
        "",
        f"## Outcome {result['outcome']}",
        "",
        f"**{result['decision']}**",
        "",
        "| Gate | Tested | Triggered |",
        "|---|---:|---:|",
    ]
    for name, gate in gates.items():
        rows.append(
            f"| {name} | {'yes' if gate.get('tested', True) else 'no'} | "
            f"{'yes' if gate.get('triggered', False) else 'no'} |"
        )
    rows.extend(
        [
            "",
            "## Decision rule",
            "",
            "A measured B6 enumeration limit is not by itself permission to add RL. Gate E is ",
            "triggered for RL only when the pre-RL mathematical approximation also misses the ",
            "frozen latency or quality threshold.",
            "",
            "## Claim boundary",
            "",
            result["claim_boundary"],
            "",
            "Machine-readable evidence: `artifacts/rl_gate_report_v2.json`.",
            "",
        ]
    )
    document.write_text("\n".join(rows), encoding="utf-8")
    print(output)
    return output


if __name__ == "__main__":
    run()
