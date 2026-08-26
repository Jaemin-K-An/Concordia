from __future__ import annotations

import hashlib
from typing import Iterable


def racing_seed(
    state_id: str,
    action_id: str,
    stage: str,
    replica: int,
    *,
    floor: int = 100000,
    ceiling: int = 999999,
) -> int:
    if replica < 0 or floor >= ceiling:
        raise ValueError("invalid racing seed parameters")
    payload = f"{state_id}|{action_id}|{stage}|{replica}".encode()
    return floor + int(hashlib.sha256(payload).hexdigest()[:16], 16) % (ceiling - floor + 1)


def assert_seed_contract(
    decision_seeds: Iterable[int],
    evaluation_seeds: Iterable[int],
) -> None:
    decision = list(map(int, decision_seeds))
    evaluation = set(map(int, evaluation_seeds))
    if len(decision) != len(set(decision)):
        raise RuntimeError("decision rollout seeds are not unique")
    overlap = set(decision) & evaluation
    if overlap:
        raise RuntimeError(f"decision/evaluation seed overlap: {sorted(overlap)}")

