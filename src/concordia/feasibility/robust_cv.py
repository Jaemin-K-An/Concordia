from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np

from concordia.feasibility.calibration import wilson_interval


def precision_constrained_threshold(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    precision_target: float,
    coverage_guard: float,
    thresholds: Sequence[float],
    eligible: Sequence[bool] | None = None,
) -> dict:
    y = np.asarray(labels, dtype=int)
    values = np.asarray(scores, dtype=float)
    allowed = np.ones(len(y), dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool)
    curve = []
    for threshold in sorted(set(float(item) for item in thresholds)):
        selected = (values >= threshold) & allowed
        count = int(selected.sum())
        success = int(y[selected].sum())
        precision = success / max(1, count)
        coverage = count / max(1, len(y))
        lower, upper = wilson_interval(success, count)
        curve.append(
            {
                "threshold": threshold,
                "intervention_count": count,
                "success_count": success,
                "precision": precision,
                "precision_ci95": [lower, upper],
                "coverage": coverage,
                "meets_precision": precision >= precision_target,
                "meets_coverage_guard": coverage >= coverage_guard,
            }
        )
    feasible = [
        row for row in curve if row["meets_precision"] and row["meets_coverage_guard"]
    ]
    if feasible:
        selected = max(
            feasible,
            key=lambda row: (row["coverage"], row["precision"], -row["threshold"]),
        )
    else:
        guarded = [row for row in curve if row["meets_coverage_guard"]]
        selected = max(
            guarded or curve,
            key=lambda row: (
                row["precision"],
                row["coverage"],
                row["intervention_count"],
                -row["threshold"],
            ),
        )
    return {"selected": selected, "curve": curve, "feasible": bool(feasible)}


def group_metrics(
    rows: Sequence[Mapping[str, object]],
    labels: Sequence[int],
    interventions: Sequence[bool],
) -> dict:
    y = np.asarray(labels, dtype=int)
    selected = np.asarray(interventions, dtype=bool)
    groupers = {
        "scenario": [str(row["scenario"]) for row in rows],
        "demand_band": [
            "low"
            if float(row["condition"]["demand_scale"]) < 0.9
            else "high"
            if float(row["condition"]["demand_scale"]) > 1.3
            else "middle"
            for row in rows
        ],
        "heterogeneity": [str(row["condition"]["heterogeneity"]) for row in rows],
        "penetration": [str(row["condition"]["navigation_penetration"]) for row in rows],
        "topology_family": [
            "distributed" if row["scenario"] in {"two_route", "ring"} else "constrained"
            for row in rows
        ],
    }
    output = {}
    precisions = []
    for dimension, values in groupers.items():
        grouped = defaultdict(list)
        for index, value in enumerate(values):
            grouped[value].append(index)
        dimension_rows = {}
        for group, indices in sorted(grouped.items()):
            index_mask = np.zeros(len(rows), dtype=bool)
            index_mask[indices] = True
            mask = index_mask & selected
            count = int(mask.sum())
            precision = float(y[mask].mean()) if count else None
            dimension_rows[group] = {
                "case_count": len(indices),
                "intervention_count": count,
                "precision": precision,
                "coverage": count / max(1, len(indices)),
            }
            if precision is not None:
                precisions.append(precision)
        output[dimension] = dimension_rows
    return {
        "dimensions": output,
        "worst_group_precision": float(min(precisions)) if precisions else 0.0,
        "median_group_precision": float(np.median(precisions)) if precisions else 0.0,
    }


def leave_group_out_folds(rows: Sequence[Mapping[str, object]]) -> list[dict]:
    definitions = {
        "CV-A_seed": [str(row["seed"]) for row in rows],
        "CV-B_demand_band": [
            "low"
            if float(row["condition"]["demand_scale"]) < 0.9
            else "high"
            if float(row["condition"]["demand_scale"]) > 1.3
            else "middle"
            for row in rows
        ],
        "CV-C_preference": [str(row["condition"]["heterogeneity"]) for row in rows],
        "CV-D_scenario": [str(row["scenario"]) for row in rows],
        "CV-E_topology": [
            "distributed" if row["scenario"] in {"two_route", "ring"} else "constrained"
            for row in rows
        ],
    }
    folds = []
    for family, groups in definitions.items():
        group_values = np.asarray(groups)
        for group in sorted(set(groups)):
            test = np.flatnonzero(group_values == group)
            train = np.flatnonzero(group_values != group)
            folds.append(
                {
                    "family": family,
                    "group": group,
                    "train_indices": train.tolist(),
                    "test_indices": test.tolist(),
                }
            )
    return folds
