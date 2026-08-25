from __future__ import annotations

from enum import Enum


class AlignmentLabel(str, Enum):
    WIN = "WIN"
    TRADEOFF = "TRADEOFF"
    INFEASIBLE = "INFEASIBLE"


def classify_alignment_case(
    *,
    baseline_ttt: float,
    adaptive_ttt: float,
    maximum_regret: float,
    regret_limit: float,
    baseline_risk: float,
    adaptive_risk: float,
    safety_delta: float,
    legal: bool,
    meaningful_intervention: bool,
    minimum_relative_ttt_gain: float,
) -> AlignmentLabel:
    feasible = (
        legal
        and maximum_regret <= regret_limit + 1e-10
        and adaptive_risk <= baseline_risk + safety_delta + 1e-10
    )
    if not feasible or not meaningful_intervention:
        return AlignmentLabel.INFEASIBLE
    gain = (baseline_ttt - adaptive_ttt) / max(baseline_ttt, 1e-12)
    return (
        AlignmentLabel.WIN
        if gain >= minimum_relative_ttt_gain
        else AlignmentLabel.TRADEOFF
    )
