from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class RobustActionOptimizer:
    minimum_benefit: float = 0.005
    unsafe_probability_threshold: float = 0.05
    maximum_regret: float = 0.08

    def select(self, candidates: Sequence[Mapping[str, object]]) -> dict:
        safe = []
        null = None
        for candidate in candidates:
            if bool(candidate.get("is_null", False)):
                null = dict(candidate)
                continue
            combined_unsafe = max(
                float(candidate["ml_unsafe_probability"]),
                float(candidate["rollout_unsafe_probability"]),
            )
            if (
                combined_unsafe <= self.unsafe_probability_threshold
                and float(candidate["predicted_max_regret"]) <= self.maximum_regret
                and bool(candidate.get("legal", True))
                and float(candidate["robust_benefit"]) > self.minimum_benefit
            ):
                safe.append((dict(candidate), combined_unsafe))
        if safe:
            selected, combined = max(
                safe,
                key=lambda pair: (float(pair[0]["robust_benefit"]), -pair[1], str(pair[0]["action_id"])),
            )
            selected["combined_unsafe_probability"] = combined
            selected["intervene"] = True
            selected["reason"] = "maximum_safe_robust_value"
            return selected
        fallback = null or {
            "action_id": "A00_NULL_B1",
            "is_null": True,
            "robust_benefit": 0.0,
        }
        fallback = dict(fallback)
        fallback["intervene"] = False
        fallback["reason"] = "all_actions_failed_robust_constraints"
        return fallback

