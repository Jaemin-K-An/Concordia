from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from concordia.selective.decision import SelectiveDecision, SelectiveOutcome


@dataclass(frozen=True)
class V5DecisionInputs:
    case_id: str
    regime: str
    shift_class: str
    domain_shift_score: float
    success_probability: float
    analytical_benefit: float
    corrected_microscopic_benefit: float
    microscopic_success_probability: float
    microscopic_safety_probability_upper: float
    legal: bool


@dataclass(frozen=True)
class RegimeConditionedPolicy:
    probability_thresholds: Mapping[str, Mapping[str, float | None]]
    shift_probability_penalty: float
    micro_success_threshold: float
    micro_safety_threshold: float
    use_shift_gate: bool = True
    use_micro_correction: bool = True
    use_micro_safety_veto: bool = True

    def decide(self, inputs: V5DecisionInputs) -> SelectiveDecision:
        regime_thresholds = self.probability_thresholds.get(inputs.regime, {})
        threshold = regime_thresholds.get(inputs.shift_class)
        failures = []
        if self.use_shift_gate and inputs.shift_class == "STRONG_SHIFT":
            failures.append("strong domain shift requires baseline fallback")
        if threshold is None:
            failures.append("regime/shift operating cell is frozen to abstain")
            threshold_value = 1.0
        else:
            threshold_value = float(threshold)
            if self.use_shift_gate:
                threshold_value = min(
                    0.999,
                    threshold_value
                    + self.shift_probability_penalty * inputs.domain_shift_score,
                )
            if inputs.success_probability < threshold_value:
                failures.append("regime-conditioned success probability below threshold")
        if self.use_micro_correction:
            if inputs.corrected_microscopic_benefit <= 0.0:
                failures.append("microscopic-corrected benefit is nonpositive")
            if inputs.microscopic_success_probability < self.micro_success_threshold:
                failures.append("microscopic success probability below threshold")
        if (
            self.use_micro_safety_veto
            and inputs.microscopic_safety_probability_upper >= self.micro_safety_threshold
        ):
            failures.append("microscopic safety veto")
        if not inputs.legal:
            failures.append("route set is illegal or non-executable")
        intervene = not failures
        return SelectiveDecision(
            case_id=inputs.case_id,
            intervene=intervene,
            outcome=SelectiveOutcome.FAILURE if intervene else SelectiveOutcome.ABSTAIN,
            selected_policy="CONCORDIA_V5" if intervene else "B1_ETA_BASELINE",
            p_win=inputs.success_probability,
            p_win_lower=inputs.success_probability,
            uncertainty=max(0.0, inputs.domain_shift_score),
            frozen_threshold=threshold_value,
            reasons=tuple(failures),
            explanation=(
                f"regime={inputs.regime}",
                f"shift={inputs.shift_class} DSS={inputs.domain_shift_score:.3f}",
                f"P(success)={inputs.success_probability:.3f}",
                f"corrected micro benefit={inputs.corrected_microscopic_benefit:.4f}",
                f"micro safety UCB={inputs.microscopic_safety_probability_upper:.3f}",
            ),
            computed_values={
                "domain_shift_score": inputs.domain_shift_score,
                "analytical_benefit": inputs.analytical_benefit,
                "corrected_microscopic_benefit": inputs.corrected_microscopic_benefit,
                "microscopic_success_probability": inputs.microscopic_success_probability,
                "microscopic_safety_probability_upper": inputs.microscopic_safety_probability_upper,
            },
        )
