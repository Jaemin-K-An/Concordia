from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np

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
    median_ttc: Optional[float]
    p90_drac: float
    p95_drac: float
    p99_drac: float
    high_closing_speed_conflicts: int
    observation_count: int


@dataclass(frozen=True)
class SafetyNonDegradationResult:
    passed: bool
    mean_risk_difference: float
    cvar_difference: float
    conflict_rate_difference: float
    delta: float
    reasons: Sequence[str]


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
    high_closing_speed_threshold: float = 10.0,
) -> SafetySummary:
    if ttc_threshold <= 0 or hard_braking_threshold >= 0 or high_closing_speed_threshold <= 0:
        raise ValidationError("invalid safety thresholds")
    ttc: List[float] = []
    drac: List[float] = []
    hard_braking = 0
    high_closing_speed = 0
    observation_count = 0
    for frame in frames:
        observation_count += 1
        if frame.follower_acceleration <= hard_braking_threshold:
            hard_braking += 1
        if frame.leader_id is None or frame.gap is None or frame.leader_speed is None:
            continue
        closing_speed = frame.follower_speed - frame.leader_speed
        if closing_speed > 0:
            if closing_speed >= high_closing_speed_threshold:
                high_closing_speed += 1
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
        median_ttc=float(np.median(ttc)) if ttc else None,
        p90_drac=float(np.percentile(drac, 90)) if drac else 0.0,
        p95_drac=float(np.percentile(drac, 95)) if drac else 0.0,
        p99_drac=float(np.percentile(drac, 99)) if drac else 0.0,
        high_closing_speed_conflicts=high_closing_speed,
        observation_count=observation_count,
    )


def safety_non_degradation(
    baseline: SafetySummary,
    proposed: SafetySummary,
    delta: float = 0.0,
) -> SafetyNonDegradationResult:
    """Require non-inferiority in mean/tail DRAC and TTC conflict rate."""
    if delta < 0:
        raise ValidationError("safety non-degradation delta cannot be negative")
    baseline_mean = float(np.mean(baseline.drac_values)) if baseline.drac_values else 0.0
    proposed_mean = float(np.mean(proposed.drac_values)) if proposed.drac_values else 0.0
    baseline_rate = baseline.ttc_conflicts / max(1, baseline.observation_count)
    proposed_rate = proposed.ttc_conflicts / max(1, proposed.observation_count)
    differences = (
        proposed_mean - baseline_mean,
        proposed.cvar_drac_95 - baseline.cvar_drac_95,
        proposed_rate - baseline_rate,
    )
    labels = ("mean_drac", "cvar_drac_95", "ttc_conflict_rate")
    reasons = tuple(label for label, difference in zip(labels, differences) if difference > delta)
    return SafetyNonDegradationResult(
        passed=not reasons,
        mean_risk_difference=differences[0],
        cvar_difference=differences[1],
        conflict_rate_difference=differences[2],
        delta=delta,
        reasons=reasons,
    )
