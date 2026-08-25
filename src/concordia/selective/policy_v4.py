from __future__ import annotations

from dataclasses import dataclass

from concordia.selective.decision import SelectiveDecision, SelectiveOutcome


@dataclass(frozen=True)
class V4DecisionInputs:
    case_id: str
    success_probability: float
    success_probability_lower: float
    expected_benefit: float
    benefit_lower: float
    safety_difference_upper: float
    safety_failure_probability: float
    safety_failure_probability_upper: float
    esiv: float
    esiv_lower: float
    legal: bool


@dataclass(frozen=True)
class PrecisionConstrainedPolicy:
    policy_name: str
    probability_threshold: float
    benefit_threshold: float
    safety_delta: float
    safety_probability_threshold: float
    esiv_threshold: float = 0.0
    use_esiv: bool = False

    def decide(self, inputs: V4DecisionInputs) -> SelectiveDecision:
        if self.use_esiv:
            primary_checks = (
                (inputs.esiv_lower >= self.esiv_threshold, "ESIV lower bound below threshold"),
            )
        else:
            primary_checks = (
                (
                    inputs.success_probability >= self.probability_threshold,
                    "calibrated success probability below threshold",
                ),
                (inputs.benefit_lower >= self.benefit_threshold, "benefit lower bound below threshold"),
            )
        safety_checks = (
            (inputs.safety_difference_upper <= self.safety_delta, "safety UCB exceeds margin"),
            (
                inputs.safety_failure_probability_upper <= self.safety_probability_threshold,
                "safety-failure probability exceeds threshold",
            ),
            (inputs.legal, "route set is illegal or non-executable"),
        )
        failures = tuple(reason for passed, reason in (*primary_checks, *safety_checks) if not passed)
        intervene = not failures
        return SelectiveDecision(
            case_id=inputs.case_id,
            intervene=intervene,
            outcome=SelectiveOutcome.FAILURE if intervene else SelectiveOutcome.ABSTAIN,
            selected_policy="CONCORDIA_V4" if intervene else "B1_ETA_BASELINE",
            p_win=inputs.success_probability,
            p_win_lower=inputs.success_probability_lower,
            uncertainty=max(0.0, inputs.success_probability - inputs.success_probability_lower),
            frozen_threshold=self.esiv_threshold if self.use_esiv else self.probability_threshold,
            reasons=failures,
            explanation=(
                f"P(success)={inputs.success_probability:.3f}",
                f"expected benefit={inputs.expected_benefit:.4f}",
                f"benefit lower={inputs.benefit_lower:.4f}",
                f"safety failure probability={inputs.safety_failure_probability:.3f}",
                f"ESIV lower={inputs.esiv_lower:.6f}",
            ),
            computed_values={
                "expected_benefit": inputs.expected_benefit,
                "benefit_lower": inputs.benefit_lower,
                "safety_difference_upper": inputs.safety_difference_upper,
                "safety_failure_probability": inputs.safety_failure_probability,
                "safety_failure_probability_upper": inputs.safety_failure_probability_upper,
                "esiv": inputs.esiv,
                "esiv_lower": inputs.esiv_lower,
            },
        )
