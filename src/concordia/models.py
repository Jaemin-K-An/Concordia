from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Sequence, Tuple

from .errors import ValidationError


FEATURE_NAMES = ("time", "variability", "cost", "risk", "complexity", "familiarity")


@dataclass(frozen=True)
class EdgeData:
    free_flow_time: float
    capacity: float
    length: float = 1.0
    alpha: float = 0.15
    beta: float = 4.0
    variability: float = 0.0
    monetary_cost: float = 0.0
    risk: float = 0.0
    complexity: float = 0.0
    legal: bool = True

    def __post_init__(self) -> None:
        if self.free_flow_time <= 0 or self.capacity <= 0 or self.length < 0:
            raise ValidationError("edge time/capacity must be positive and length non-negative")
        if self.alpha < 0 or self.beta < 1:
            raise ValidationError("BPR alpha must be non-negative and beta >= 1")
        for name in ("variability", "monetary_cost", "risk", "complexity"):
            if getattr(self, name) < 0:
                raise ValidationError("edge attributes cannot be negative")

    def travel_time(self, flow: float) -> float:
        if flow < 0:
            raise ValidationError("traffic flow cannot be negative")
        return self.free_flow_time * (1.0 + self.alpha * (flow / self.capacity) ** self.beta)

    def derivative(self, flow: float) -> float:
        if flow < 0:
            raise ValidationError("traffic flow cannot be negative")
        if flow == 0 and self.beta > 1:
            return 0.0
        return (
            self.free_flow_time
            * self.alpha
            * self.beta
            * (flow ** (self.beta - 1))
            / (self.capacity**self.beta)
        )

    def integral(self, flow: float) -> float:
        if flow < 0:
            raise ValidationError("traffic flow cannot be negative")
        return self.free_flow_time * (
            flow + self.alpha * flow ** (self.beta + 1) / ((self.beta + 1) * self.capacity**self.beta)
        )


@dataclass(frozen=True)
class RouteFeatures:
    time: float
    variability: float = 0.0
    cost: float = 0.0
    risk: float = 0.0
    complexity: float = 0.0
    familiarity: float = 0.0

    def __post_init__(self) -> None:
        for name in FEATURE_NAMES:
            value = getattr(self, name)
            if value < 0:
                raise ValidationError(f"route feature {name} cannot be negative")
        if self.familiarity > 1:
            raise ValidationError("familiarity must be in [0, 1]")

    def as_dict(self) -> Dict[str, float]:
        return {name: float(getattr(self, name)) for name in FEATURE_NAMES}


@dataclass(frozen=True)
class Route:
    route_id: str
    nodes: Tuple[str, ...]
    features: RouteFeatures

    def __post_init__(self) -> None:
        if not self.route_id or len(self.nodes) < 2:
            raise ValidationError("route requires an id and at least two nodes")
        if len(set(zip(self.nodes, self.nodes[1:]))) != len(self.nodes) - 1:
            raise ValidationError("route cannot repeat an edge")

    @property
    def edges(self) -> Tuple[Tuple[str, str], ...]:
        return tuple(zip(self.nodes, self.nodes[1:]))


@dataclass(frozen=True)
class PreferenceVector:
    time: float
    variability: float
    cost: float
    risk: float
    complexity: float
    familiarity: float

    def __post_init__(self) -> None:
        values = [getattr(self, name) for name in FEATURE_NAMES]
        if any(value < 0 for value in values):
            raise ValidationError("preference weights cannot be negative")
        if sum(values) <= 0:
            raise ValidationError("at least one preference weight must be positive")

    def normalized(self) -> "PreferenceVector":
        total = sum(getattr(self, name) for name in FEATURE_NAMES)
        return PreferenceVector(**{name: getattr(self, name) / total for name in FEATURE_NAMES})

    def as_dict(self) -> Dict[str, float]:
        return {name: float(getattr(self, name)) for name in FEATURE_NAMES}


@dataclass(frozen=True)
class User:
    user_id: str
    origin: str
    destination: str
    preferences: PreferenceVector
    epsilon: float = 0.0
    rationality: float = 5.0

    def __post_init__(self) -> None:
        if not self.user_id or self.origin == self.destination:
            raise ValidationError("user id and distinct OD nodes are required")
        if self.epsilon < 0 or self.rationality < 0:
            raise ValidationError("epsilon and rationality must be non-negative")


@dataclass(frozen=True)
class FeatureScales:
    time: float = 30.0
    variability: float = 10.0
    cost: float = 5000.0
    risk: float = 1.0
    complexity: float = 1.0
    familiarity: float = 1.0

    def __post_init__(self) -> None:
        if any(getattr(self, name) <= 0 for name in FEATURE_NAMES):
            raise ValidationError("all feature scales must be positive")

    def as_dict(self) -> Dict[str, float]:
        return {name: float(getattr(self, name)) for name in FEATURE_NAMES}


@dataclass(frozen=True)
class AssignmentResult:
    assignments: Mapping[str, str]
    objective: float
    total_travel_time: float
    total_safety_risk: float
    total_ghost_risk: float
    route_entropy: float
    regrets: Mapping[str, float]
    metadata: Mapping[str, object] = field(default_factory=dict)


Path = Sequence[str]
EdgeKey = Tuple[str, str]
