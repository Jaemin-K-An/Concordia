from __future__ import annotations

from typing import List

import numpy as np

from concordia.errors import ValidationError
from concordia.models import FEATURE_NAMES, PreferenceVector, User


_MEAN = np.asarray([0.36, 0.16, 0.10, 0.18, 0.10, 0.10], dtype=float)


def generate_population(
    count: int,
    origin: str,
    destination: str,
    heterogeneity: str,
    epsilon: float,
    rationality: float,
    seed: int,
) -> List[User]:
    """Generate synthetic preferences with approximately fixed mean and changed variance."""
    if count < 1 or heterogeneity not in {"none", "low", "medium", "high", "bimodal", "long_tail"}:
        raise ValidationError("invalid population size or heterogeneity")
    rng = np.random.default_rng(seed)
    if heterogeneity == "none":
        samples = np.tile(_MEAN, (count, 1))
    elif heterogeneity in {"low", "medium", "high"}:
        # Antithetic pairs preserve the declared population mean exactly while changing
        # only dispersion. This supports controlled tests of the variance hypothesis.
        magnitude = {"low": 0.015, "medium": 0.045, "high": 0.085}[heterogeneity]
        rows = []
        for _ in range(count // 2):
            direction = rng.normal(size=len(FEATURE_NAMES))
            direction -= direction.mean()
            direction *= magnitude / max(float(np.abs(direction).max()), 1e-12)
            rows.extend((_MEAN + direction, _MEAN - direction))
        if count % 2:
            rows.append(_MEAN.copy())
        samples = np.vstack(rows)
    elif heterogeneity == "bimodal":
        first = _MEAN.copy()
        second = _MEAN.copy()
        shift = min(0.12, first[0], second[3])
        first[0], first[3] = first[0] + shift, first[3] - shift
        second[0], second[3] = second[0] - shift, second[3] + shift
        samples = np.vstack(
            [rng.dirichlet(first * 80.0) if index % 2 == 0 else rng.dirichlet(second * 80.0) for index in range(count)]
        )
    else:
        samples = rng.dirichlet(_MEAN * 15.0, size=count)
        tail_count = max(1, count // 10)
        samples[:tail_count] = rng.dirichlet(np.ones(len(FEATURE_NAMES)) * 0.25, size=tail_count)
    users: List[User] = []
    for index, weights in enumerate(samples):
        users.append(
            User(
                user_id=f"u{index:04d}",
                origin=origin,
                destination=destination,
                preferences=PreferenceVector(**dict(zip(FEATURE_NAMES, weights.tolist()))),
                epsilon=epsilon,
                rationality=rationality,
            )
        )
    return users
