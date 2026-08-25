from __future__ import annotations


def expected_safe_intervention_value(
    success_probability: float,
    expected_benefit: float,
    safety_failure_probability: float,
) -> float:
    probability = min(1.0, max(0.0, float(success_probability)))
    benefit = max(0.0, float(expected_benefit))
    risk = min(1.0, max(0.0, float(safety_failure_probability)))
    return probability * benefit * (1.0 - risk)


def risk_adjusted_esiv(
    success_probability_lower: float,
    benefit_lower: float,
    safety_failure_probability_upper: float,
) -> float:
    return expected_safe_intervention_value(
        success_probability_lower,
        benefit_lower,
        safety_failure_probability_upper,
    )
