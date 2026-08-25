from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping


@dataclass(frozen=True)
class SafeMicroLabel:
    safe_micro_success: bool
    diagnostic_class: str
    relative_ttt_gain: float
    safety_difference: float
    traffic_benefit_pass: bool
    safety_pass: bool
    regret_pass: bool
    legal_pass: bool

    def to_dict(self) -> dict:
        return asdict(self)


def build_safe_micro_label(
    baseline: Mapping[str, object],
    adaptive: Mapping[str, object],
    *,
    minimum_relative_ttt_gain: float,
    safety_margin: float,
    regret_limit: float,
) -> SafeMicroLabel:
    baseline_ttt = float(baseline["total_travel_time_seconds"])
    adaptive_ttt = float(adaptive["total_travel_time_seconds"])
    relative_gain = (baseline_ttt - adaptive_ttt) / max(baseline_ttt, 1e-9)
    baseline_risk = float(baseline["safety"]["cvar_drac_95"])
    adaptive_risk = float(adaptive["safety"]["cvar_drac_95"])
    safety_difference = adaptive_risk - baseline_risk
    benefit = relative_gain >= float(minimum_relative_ttt_gain)
    safety = safety_difference <= float(safety_margin)
    regret = float(adaptive["maximum_affected_regret"]) <= float(regret_limit) + 1e-10
    legal = bool(adaptive["all_executed_routes_legal"])
    success = benefit and safety and regret and legal
    if safety:
        if benefit:
            diagnostic = "S1_safe_beneficial"
        elif relative_gain > -float(minimum_relative_ttt_gain):
            diagnostic = "S2_safe_neutral"
        else:
            diagnostic = "S3_safe_harmful"
    else:
        diagnostic = "U1_unsafe_beneficial" if benefit else "U2_unsafe_harmful"
    return SafeMicroLabel(
        success,
        diagnostic,
        relative_gain,
        safety_difference,
        benefit,
        safety,
        regret,
        legal,
    )
