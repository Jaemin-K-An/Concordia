from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence


INTENSITIES = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
USER_STRATEGIES = (
    "U1_lowest_slack",
    "U2_highest_marginal_benefit",
    "U3_highest_VDE",
    "U4_ETA_sensitive",
    "U5_reliability_sensitive",
    "U6_safety_sensitive",
    "U7_highest_acceptance",
    "U8_hybrid",
)
ROUTE_ALLOCATIONS = (
    "R1_best_alternative",
    "R2_capacity_proportional",
    "R3_inverse_overlap",
    "R4_reliability_weighted",
    "R5_minimum_bottleneck_load",
    "R6_entropy_constrained",
)


@dataclass(frozen=True)
class AdaptiveAction:
    action_id: str
    reroute_fraction: float
    user_strategy: str
    route_allocation: str
    is_null: bool = False

    def __post_init__(self) -> None:
        if not self.action_id:
            raise ValueError("action_id is required")
        if self.is_null:
            if self.reroute_fraction != 0.0:
                raise ValueError("null action must have zero diversion")
            return
        if self.reroute_fraction not in INTENSITIES:
            raise ValueError("action diversion is outside the frozen initial library")
        if self.user_strategy not in USER_STRATEGIES:
            raise ValueError("unknown user-selection strategy")
        if self.route_allocation not in ROUTE_ALLOCATIONS:
            raise ValueError("unknown route-allocation strategy")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "AdaptiveAction":
        return cls(
            str(value["action_id"]),
            float(value["reroute_fraction"]),
            str(value["user_strategy"]),
            str(value["route_allocation"]),
            bool(value.get("is_null", False)),
        )


def null_action() -> AdaptiveAction:
    return AdaptiveAction("A00_NULL_B1", 0.0, "U1_lowest_slack", "R1_best_alternative", True)


def generate_action_library(count: int = 25) -> tuple[AdaptiveAction, ...]:
    """Balanced deterministic library; null plus 24 actions by preregistration."""
    if count != 25:
        raise ValueError("initial v9 library is frozen at 25 actions")
    actions = [null_action()]
    for index in range(24):
        intensity = INTENSITIES[index % len(INTENSITIES)]
        user = USER_STRATEGIES[index % len(USER_STRATEGIES)]
        allocation = ROUTE_ALLOCATIONS[(index * 5 + index // 6) % len(ROUTE_ALLOCATIONS)]
        actions.append(AdaptiveAction(f"A{index + 1:02d}", intensity, user, allocation))
    if set(action.reroute_fraction for action in actions[1:]) != set(INTENSITIES):
        raise RuntimeError("balanced action design lost an intensity")
    if set(action.user_strategy for action in actions[1:]) != set(USER_STRATEGIES):
        raise RuntimeError("balanced action design lost a user strategy")
    if set(action.route_allocation for action in actions[1:]) != set(ROUTE_ALLOCATIONS):
        raise RuntimeError("balanced action design lost an allocation rule")
    return tuple(actions)


def b6_reference_action() -> dict:
    return {
        "action_id": "B6_ALWAYS_ON_REFERENCE",
        "reroute_fraction": 1.0,
        "user_strategy": "U7_highest_acceptance",
        "route_allocation": "R1_best_alternative",
        "is_null": False,
        "reference_only": True,
    }


def validate_library(actions: Sequence[AdaptiveAction]) -> None:
    if not actions or not actions[0].is_null:
        raise ValueError("baseline null action must be first and always present")
    if len({action.action_id for action in actions}) != len(actions):
        raise ValueError("action identifiers must be unique")

