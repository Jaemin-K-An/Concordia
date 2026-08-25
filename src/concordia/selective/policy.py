from __future__ import annotations

from dataclasses import dataclass

from concordia.feasibility.gate import FeasibilityGate
from concordia.selective.decision import SelectiveDecision, SelectiveOutcome


@dataclass(frozen=True)
class SelectiveInterventionPolicy:
    gate: FeasibilityGate

    def decide(
        self,
        *,
        case_id: str,
        p_win: float,
        p_win_lower: float,
        uncertainty: float,
        alignment_potential: float,
        route_overlap: float,
        safety_upper_difference: float,
        acceptance_probability: float,
        ttt_lcb_gain: float,
        predicted_tail_loss: float,
        legal: bool,
        counterfactual_success: bool | None = None,
    ) -> SelectiveDecision:
        gate_decision = self.gate.decide(
            probability=p_win,
            probability_lower=p_win_lower,
            uncertainty=uncertainty,
            safety_upper_difference=safety_upper_difference,
            acceptance_probability=acceptance_probability,
            ttt_lcb_gain=ttt_lcb_gain,
            predicted_tail_loss=predicted_tail_loss,
            legal=legal,
        )
        outcome = (
            SelectiveOutcome.ABSTAIN
            if not gate_decision.intervene
            else SelectiveOutcome.SUCCESS
            if counterfactual_success
            else SelectiveOutcome.FAILURE
        )
        explanation = (
            f"P(WIN): {p_win:.3f} vs threshold {self.gate.probability_threshold:.3f}",
            f"P_lower(WIN): {p_win_lower:.3f}",
            f"APS: {alignment_potential:.6f}",
            f"route overlap: {route_overlap:.3f}",
            f"safety upper difference: {safety_upper_difference:.4f}",
            f"TTT LCB gain: {ttt_lcb_gain:.4f}",
        )
        return SelectiveDecision(
            case_id=case_id,
            intervene=gate_decision.intervene,
            outcome=outcome,
            selected_policy="CONCORDIA_V3" if gate_decision.intervene else "B1_ETA_BASELINE",
            p_win=p_win,
            p_win_lower=p_win_lower,
            uncertainty=uncertainty,
            frozen_threshold=self.gate.probability_threshold,
            reasons=gate_decision.reasons,
            explanation=explanation,
            computed_values={
                "alignment_potential": alignment_potential,
                "route_overlap": route_overlap,
                "safety_margin": self.gate.safety_delta - safety_upper_difference,
                "acceptance_probability": acceptance_probability,
                "ttt_lcb_gain": ttt_lcb_gain,
                "predicted_tail_loss": predicted_tail_loss,
            },
        )
