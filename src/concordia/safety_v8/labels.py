from __future__ import annotations

from typing import Mapping


SAFETY_DELTA = 0.25


def unsafe_intervention(row: Mapping[str, object], delta: float = SAFETY_DELTA) -> bool:
    """Registered paired label: Adaptive increases CVaR-DRAC by more than delta."""
    if "outcomes" in row:
        return float(row["outcomes"]["tau_s"]) > delta
    return float(row["risk_adaptive"]) > float(row["risk_b1"]) + delta


def safety_label_record(row: Mapping[str, object], delta: float = SAFETY_DELTA) -> dict:
    return {
        "pair_id": str(row["pair_id"]),
        "unsafe_intervention": int(unsafe_intervention(row, delta)),
        "risk_b1": float(row["risk_b1"]),
        "risk_adaptive": float(row["risk_adaptive"]),
        "safety_delta": float(delta),
        "label_definition": "risk_adaptive > risk_b1 + safety_delta",
    }

