from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from concordia.errors import ValidationError


@dataclass(frozen=True)
class TrajectoryFrame:
    time: float
    follower_id: str
    leader_id: Optional[str]
    gap: Optional[float]
    follower_speed: float
    leader_speed: Optional[float]
    follower_acceleration: float = 0.0

    def __post_init__(self) -> None:
        if self.time < 0 or self.follower_speed < 0:
            raise ValidationError("trajectory time and speed cannot be negative")
        if self.gap is not None and self.gap <= 0:
            raise ValidationError("recorded following gap must be positive")
        if self.leader_speed is not None and self.leader_speed < 0:
            raise ValidationError("leader speed cannot be negative")


@dataclass(frozen=True)
class SafetySummary:
    ttc_values: Sequence[float]
    drac_values: Sequence[float]
    pet_values: Sequence[float]
    ttc_conflicts: int
    hard_braking_events: int
    cvar_drac_95: float
    min_ttc: Optional[float]


def _cvar_upper(values: Sequence[float], alpha: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    start = min(len(ordered) - 1, math.floor(alpha * len(ordered)))
    tail = ordered[start:]
    return sum(tail) / len(tail)


def summarize_safety(
    frames: Iterable[TrajectoryFrame],
    pet_values: Iterable[float] = (),
    ttc_threshold: float = 1.5,
    hard_braking_threshold: float = -4.5,
) -> SafetySummary:
    if ttc_threshold <= 0 or hard_braking_threshold >= 0:
        raise ValidationError("invalid safety thresholds")
    ttc: List[float] = []
    drac: List[float] = []
    hard_braking = 0
    for frame in frames:
        if frame.follower_acceleration <= hard_braking_threshold:
            hard_braking += 1
        if frame.leader_id is None or frame.gap is None or frame.leader_speed is None:
            continue
        closing_speed = frame.follower_speed - frame.leader_speed
        if closing_speed > 0:
            ttc.append(frame.gap / closing_speed)
            drac.append(closing_speed**2 / (2.0 * frame.gap))
    pets = [float(value) for value in pet_values]
    if any(value < 0 for value in pets):
        raise ValidationError("PET values cannot be negative")
    return SafetySummary(
        ttc_values=tuple(ttc),
        drac_values=tuple(drac),
        pet_values=tuple(pets),
        ttc_conflicts=sum(value < ttc_threshold for value in ttc),
        hard_braking_events=hard_braking,
        cvar_drac_95=_cvar_upper(drac, 0.95),
        min_ttc=min(ttc) if ttc else None,
    )
