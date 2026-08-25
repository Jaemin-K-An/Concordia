from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GateDecision:
    intervene: bool
    decision: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class FeasibilityGate:
    probability_threshold: float
    maximum_uncertainty: float
    safety_delta: float
    minimum_acceptance_probability: float
    minimum_ttt_lcb_gain: float
    maximum_tail_loss: float

    def decide(
        self,
        *,
        probability: float,
        probability_lower: float,
        uncertainty: float,
        safety_upper_difference: float,
        acceptance_probability: float,
        ttt_lcb_gain: float,
        predicted_tail_loss: float,
        legal: bool,
    ) -> GateDecision:
        checks = (
            (probability >= self.probability_threshold, "P(WIN) below frozen threshold"),
            (probability_lower > 0.5, "lower WIN probability is not above 0.5"),
            (uncertainty <= self.maximum_uncertainty, "model uncertainty too high"),
            (safety_upper_difference <= self.safety_delta, "safety hard gate failed"),
            (
                acceptance_probability >= self.minimum_acceptance_probability,
                "acceptance probability too low",
            ),
            (ttt_lcb_gain >= self.minimum_ttt_lcb_gain, "TTT lower-confidence gain too low"),
            (predicted_tail_loss <= self.maximum_tail_loss, "tail-loss gate failed"),
            (legal, "route set is not legal/connected"),
        )
        failures = tuple(reason for passed, reason in checks if not passed)
        return GateDecision(not failures, "INTERVENE" if not failures else "ABSTAIN", failures)
