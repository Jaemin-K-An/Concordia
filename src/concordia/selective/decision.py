from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class SelectiveOutcome(str, Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    ABSTAIN = "ABSTAIN_BASELINE_FALLBACK"


@dataclass(frozen=True)
class SelectiveDecision:
    case_id: str
    intervene: bool
    outcome: SelectiveOutcome
    selected_policy: str
    p_win: float
    p_win_lower: float
    uncertainty: float
    frozen_threshold: float
    reasons: tuple[str, ...]
    explanation: tuple[str, ...]
    computed_values: Mapping[str, float]
