from __future__ import annotations

import hashlib
from typing import Iterable


def rollout_seed(state_id: str, action_id: str, replica: int, floor: int = 20000) -> int:
    payload = f"{state_id}|{action_id}|{replica}".encode()
    return floor + int(hashlib.sha256(payload).hexdigest()[:12], 16) % 100000


def rollout_seeds(
    state_id: str,
    action_id: str,
    replicas: int,
    realized_seed: int,
) -> tuple[int, ...]:
    values = tuple(rollout_seed(state_id, action_id, replica) for replica in range(replicas))
    if len(set(values)) != len(values) or realized_seed in values:
        raise RuntimeError("rollout and realized evaluation seeds must be disjoint")
    return values


def assert_seed_disjoint(realized_seeds: Iterable[int], decision_rollout_seeds: Iterable[int]) -> None:
    overlap = set(map(int, realized_seeds)) & set(map(int, decision_rollout_seeds))
    if overlap:
        raise RuntimeError(f"rollout/evaluation seed overlap: {sorted(overlap)}")

