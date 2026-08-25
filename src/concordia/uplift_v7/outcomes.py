from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping


@dataclass(frozen=True)
class PairedTreatmentOutcomes:
    tau_t_seconds: float
    tau_t_relative: float
    tau_s: float
    max_regret: float
    legal: bool
    safe_micro_success: bool
    benefit_magnitude_bin: str

    def to_dict(self) -> dict:
        return asdict(self)


def paired_treatment_outcomes(
    baseline: Mapping[str, object],
    adaptive: Mapping[str, object],
    *,
    minimum_relative_uplift: float = 0.01,
    safety_margin: float = 0.25,
    regret_limit: float = 0.08,
) -> PairedTreatmentOutcomes:
    ttt_b1 = float(baseline["total_travel_time_seconds"])
    ttt_adaptive = float(adaptive["total_travel_time_seconds"])
    tau_t = ttt_b1 - ttt_adaptive
    tau_t_relative = tau_t / max(ttt_b1, 1e-9)
    risk_b1 = float(baseline["safety"]["cvar_drac_95"])
    risk_adaptive = float(adaptive["safety"]["cvar_drac_95"])
    tau_s = risk_adaptive - risk_b1
    regret = float(adaptive["maximum_affected_regret"])
    legal = bool(adaptive["all_executed_routes_legal"])
    success = bool(
        tau_t_relative > minimum_relative_uplift
        and tau_s <= safety_margin
        and regret <= regret_limit
        and legal
    )
    if tau_t_relative <= minimum_relative_uplift:
        magnitude = "non_positive_or_below_minimum"
    elif tau_t_relative < 0.02:
        magnitude = "weak_1_to_2_percent"
    elif tau_t_relative < 0.05:
        magnitude = "moderate_2_to_5_percent"
    else:
        magnitude = "strong_above_5_percent"
    return PairedTreatmentOutcomes(
        tau_t,
        tau_t_relative,
        tau_s,
        regret,
        legal,
        success,
        magnitude,
    )
