from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def within_state_ranking_metrics(
    rows: Sequence[Mapping[str, object]], predictions: Sequence[float], top_m: int = 5
) -> dict:
    grouped: dict[str, list[tuple[Mapping[str, object], float]]] = {}
    for row, prediction in zip(rows, predictions):
        grouped.setdefault(str(row["state_id"]), []).append((row, float(prediction)))
    top1 = topm = 0
    ndcg_values = []
    action_regret = []
    for values in grouped.values():
        actual = np.asarray([float(row["outcomes"]["tau_t_relative"]) for row, _ in values])
        predicted = np.asarray([prediction for _, prediction in values])
        feasible = np.asarray([
            float(row["outcomes"]["tau_s"]) <= 0.25
            and float(row["outcomes"]["max_regret"]) <= 0.08
            and bool(row["outcomes"]["legal"])
            for row, _ in values
        ])
        safe_value = np.where(feasible, actual, -1e9)
        oracle = int(np.argmax(safe_value))
        oracle_value = float(safe_value[oracle])
        oracle_set = set(np.flatnonzero(np.isclose(safe_value, oracle_value, atol=1e-12)))
        order = np.argsort(-predicted, kind="stable")
        top1 += int(int(order[0]) in oracle_set)
        topm += int(bool(oracle_set.intersection(map(int, order[: min(top_m, len(order))]))))
        gain = np.maximum(safe_value, 0.0)
        discount = 1.0 / np.log2(np.arange(2, len(gain) + 2))
        dcg = float((gain[order] * discount).sum())
        idcg = float((np.sort(gain)[::-1] * discount).sum())
        ndcg_values.append(dcg / idcg if idcg else 0.0)
        action_regret.append(float(max(0.0, oracle_value - actual[order[0]])))
    count = len(grouped)
    return {
        "state_count": count,
        "top1_action_accuracy": top1 / count if count else 0.0,
        "top_5_oracle_recall": topm / count if count else 0.0,
        "ndcg_action": float(np.mean(ndcg_values)) if ndcg_values else 0.0,
        "mean_action_regret": float(np.mean(action_regret)) if action_regret else 0.0,
    }
