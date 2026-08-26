from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def _normalize(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    scale = float(np.ptp(array))
    return (array - float(array.min())) / scale if scale > 1e-12 else np.zeros(len(array))


def preference_feasible(records: Sequence[Mapping[str, float]]) -> list[dict]:
    return [dict(record) for record in records if float(record["preference_slack"]) <= float(record["epsilon"]) + 1e-12]


def rank_feasible_users(records: Sequence[Mapping[str, float]], strategy: str) -> list[dict]:
    feasible = preference_feasible(records)
    if not feasible:
        return []
    slack = _normalize([float(row["preference_slack"]) for row in feasible])
    benefit = _normalize([float(row["marginal_network_benefit"]) for row in feasible])
    acceptance = _normalize([float(row["acceptance_probability"]) for row in feasible])
    exposure = _normalize([float(row["safety_exposure"]) for row in feasible])
    for index, row in enumerate(feasible):
        if strategy == "U1_lowest_slack":
            score = -slack[index]
        elif strategy == "U2_highest_marginal_benefit":
            score = benefit[index]
        elif strategy == "U3_highest_VDE":
            score = benefit[index] / max(slack[index] + 0.05, 0.05)
        elif strategy == "U4_ETA_sensitive":
            score = float(row["weight_time"])
        elif strategy == "U5_reliability_sensitive":
            score = float(row["weight_reliability"])
        elif strategy == "U6_safety_sensitive":
            score = float(row["weight_safety"])
        elif strategy == "U7_highest_acceptance":
            score = acceptance[index]
        elif strategy == "U8_hybrid":
            score = 0.45 * benefit[index] - 0.25 * slack[index] + 0.25 * acceptance[index] - 0.20 * exposure[index]
        else:
            raise ValueError(f"unknown user strategy: {strategy}")
        row["selection_score"] = float(score)
    return sorted(feasible, key=lambda row: (-float(row["selection_score"]), str(row["vehicle_id"])))


def select_user_ids(records: Sequence[Mapping[str, float]], strategy: str, count: int) -> tuple[str, ...]:
    ranked = rank_feasible_users(records, strategy)
    selected = ranked[: max(0, count)]
    if any(float(row["preference_slack"]) > float(row["epsilon"]) + 1e-12 for row in selected):
        raise RuntimeError("Preference-Slack-violating user entered an action")
    return tuple(str(row["vehicle_id"]) for row in selected)

