from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from concordia.errors import ValidationError


GATE_THRESHOLDS = {
    "performance_gap_relative": 0.05,
    "performance_gap_scenario_families": 2,
    "runtime_cycle_seconds": 5.0,
    "runtime_exceedance_fraction": 0.20,
    "dynamic_degradation_relative": 0.10,
    "oscillation_switches_per_user_step": 0.25,
    "scalability_cycle_seconds": 5.0,
}


def evaluate_rl_gate(research: Mapping[str, Any]) -> Dict[str, Any]:
    focused = research.get("focused")
    if not isinstance(focused, list) or not focused:
        raise ValidationError("RL gate requires completed focused baseline rows")
    gaps_by_scenario: Dict[str, list[float]] = {}
    runtimes = []
    reversals_by_scenario: Dict[str, list[float]] = {}
    for row in focused:
        policies = row["policies"]
        feasible_reference = policies["B4"]["total_travel_time_vehicle_minutes_per_hour"]
        mpc = policies["B6"]["total_travel_time_vehicle_minutes_per_hour"]
        gaps_by_scenario.setdefault(row["scenario"], []).append(
            (mpc - feasible_reference) / feasible_reference
        )
        runtimes.append(float(policies["B6"]["latency_seconds"]))
        reversals_by_scenario.setdefault(row["scenario"], []).append(
            float(policies["B6"].get("route_reversal_count", 0.0))
            / (3.0 * len(policies["B6"]["assignments"]))
        )
    scenario_gaps = {
        scenario: float(np.median(values)) for scenario, values in gaps_by_scenario.items()
    }
    persistent_gap_families = sum(
        gap > GATE_THRESHOLDS["performance_gap_relative"]
        for gap in scenario_gaps.values()
    )
    runtime_exceedance = float(
        np.mean(np.asarray(runtimes) > GATE_THRESHOLDS["runtime_cycle_seconds"])
    )
    scenario_reversal_rates = {
        scenario: float(np.median(values))
        for scenario, values in reversals_by_scenario.items()
    }
    unstable_families = sum(
        value > GATE_THRESHOLDS["oscillation_switches_per_user_step"]
        for value in scenario_reversal_rates.values()
    )
    gates = {
        "A_performance_gap": {
            "tested": True,
            "threshold": (
                ">5% median B6 disadvantage against regret-feasible B4 in at least 2 "
                "scenario families"
            ),
            "scenario_median_relative_gaps": scenario_gaps,
            "qualifying_scenario_families": persistent_gap_families,
            "triggered": persistent_gap_families
            >= GATE_THRESHOLDS["performance_gap_scenario_families"],
        },
        "B_runtime": {
            "tested": True,
            "threshold": ">5 seconds in more than 20% of focused recommendation cycles",
            "p95_end_to_end_seconds": float(np.percentile(runtimes, 95)),
            "maximum_end_to_end_seconds": float(np.max(runtimes)),
            "exceedance_fraction": runtime_exceedance,
            "triggered": runtime_exceedance > GATE_THRESHOLDS["runtime_exceedance_fraction"],
        },
        "C_dynamic_generalization": {
            "tested": False,
            "threshold": ">10% degradation under a pre-registered nonstationary environment",
            "triggered": False,
            "reason": "no trained policy exists and drift evidence is unit-level, not a full traffic study",
        },
        "D_feedback_instability": {
            "tested": True,
            "threshold": ">0.25 back-and-forth route reversals per user per step in at least 2 families",
            "scenario_median_reversal_rates": scenario_reversal_rates,
            "qualifying_scenario_families": unstable_families,
            "triggered": unstable_families >= 2,
            "reason": "accepted one-way route changes are not counted as feedback oscillation",
        },
        "E_scalability": {
            "tested": False,
            "threshold": ">5 seconds at the declared operational scale",
            "triggered": False,
            "reason": "focused study is a 6-user correctness scale; large-scale solver evidence is absent",
        },
    }
    triggered = [name for name, result in gates.items() if result["triggered"]]
    passed = bool(triggered)
    return {
        "outcome": "GATE_PASSED_REQUIRES_RL" if passed else "A",
        "decision": (
            "RL research is authorized because at least one quantitative gate passed."
            if passed
            else "RL not introduced because deterministic/receding-horizon optimization was "
            "sufficient under the tested, declared small-instance conditions."
        ),
        "rl_authorized": passed,
        "rl_introduced": False,
        "thresholds_frozen_before_decision": GATE_THRESHOLDS,
        "gates": gates,
        "triggered_gates": triggered,
        "rationale": (
            f"Quantitative gate(s) passed: {', '.join(triggered)}."
            if passed
            else "No quantitatively demonstrated unsolved problem passed a gate. Untested dynamic "
            "generalization and large-scale behavior are limitations, not evidence for RL."
        ),
        "claim_boundary": (
            "This decision applies only to the tested analytical correctness scale and does not "
            "claim that RL can never help on larger or nonstationary networks."
        ),
    }


def write_rl_gate(result: Mapping[str, Any], json_path: str, document_path: str) -> None:
    json_destination = Path(json_path)
    json_destination.parent.mkdir(parents=True, exist_ok=True)
    json_destination.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# RL gate decision",
        "",
        f"## Outcome {result['outcome']}",
        "",
        f"**{result['decision']}**",
        "",
        result["rationale"],
        "",
        "| Gate | Tested | Triggered | Evidence |",
        "|---|---:|---:|---|",
    ]
    for name, gate in result["gates"].items():
        evidence = gate.get("reason", gate.get("threshold", ""))
        lines.append(
            f"| {name} | {'yes' if gate['tested'] else 'no'} | "
            f"{'yes' if gate['triggered'] else 'no'} | {evidence} |"
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            str(result["claim_boundary"]),
            "",
            "The machine-readable evidence is `artifacts/rl_gate_report.json`.",
            "",
        ]
    )
    Path(document_path).write_text("\n".join(lines), encoding="utf-8")
