from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence


def action_outcome(row: Mapping[str, object], minimum_benefit: float = 0.005) -> dict:
    traffic = float(row["outcomes"]["tau_t_relative"])
    safety = float(row["outcomes"]["tau_s"])
    regret = float(row["outcomes"]["max_regret"])
    legal = bool(row["outcomes"]["legal"])
    safety_pass = safety <= 0.25
    regret_pass = regret <= 0.08
    safe_beneficial = bool(traffic > minimum_benefit and safety_pass and regret_pass and legal)
    return {
        "traffic_benefit": traffic,
        "safety_pass": safety_pass,
        "regret_pass": regret_pass,
        "legal": legal,
        "safe_beneficial": safe_beneficial,
    }


def oracle_for_state(rows: Sequence[Mapping[str, object]]) -> dict:
    if not rows:
        raise ValueError("oracle state has no evaluated actions")
    enriched = [(row, action_outcome(row)) for row in rows]
    feasible = [pair for pair in enriched if pair[1]["safety_pass"] and pair[1]["regret_pass"] and pair[1]["legal"]]
    best = max(feasible, key=lambda pair: pair[1]["traffic_benefit"]) if feasible else None
    beneficial = [pair for pair in feasible if pair[1]["safe_beneficial"]]
    return {
        "state_id": str(rows[0]["state_id"]),
        "evaluated_action_count": len(rows),
        "safe_beneficial_action_exists": bool(beneficial),
        "safe_beneficial_action_count": len(beneficial),
        "oracle_action_id": str(best[0]["action_id"]) if best else "A00_NULL_B1",
        "oracle_benefit": float(best[1]["traffic_benefit"]) if best else 0.0,
        "oracle_safe_beneficial_action_id": str(max(beneficial, key=lambda pair: pair[1]["traffic_benefit"])[0]["action_id"]) if beneficial else None,
        "oracle_safe_benefit": float(max(pair[1]["traffic_benefit"] for pair in beneficial)) if beneficial else 0.0,
    }


def oracle_actionability(rows: Sequence[Mapping[str, object]]) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["state_id"])].append(row)
    states = [oracle_for_state(values) for _, values in sorted(grouped.items())]
    actionable = sum(bool(state["safe_beneficial_action_exists"]) for state in states)
    return {
        "state_count": len(states),
        "actionable_state_count": actionable,
        "oracle_actionability_rate": actionable / len(states) if states else 0.0,
        "states": states,
    }

