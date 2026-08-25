from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from concordia.feasibility.calibration import wilson_interval


def summarize_selective_policy(rows: Sequence[Mapping[str, object]]) -> dict:
    count = len(rows)
    interventions = [row for row in rows if bool(row["intervene"])]
    successes = [row for row in interventions if bool(row["success"])]
    adaptive_failures = [row for row in rows if not bool(row["counterfactual_success"])]
    wins = [row for row in rows if bool(row["counterfactual_success"])]
    abstained_failures = [row for row in adaptive_failures if not bool(row["intervene"])]
    missed = [row for row in wins if not bool(row["intervene"])]
    precision = len(successes) / max(1, len(interventions))
    coverage = len(interventions) / max(1, count)
    pbr = len(successes) / max(1, count)
    precision_ci = wilson_interval(len(successes), len(interventions))
    coverage_ci = wilson_interval(len(interventions), count)
    gains = np.asarray([float(row["system_ttt_gain"]) for row in rows], dtype=float)
    return {
        "case_count": count,
        "intervention_count": len(interventions),
        "successful_intervention_count": len(successes),
        "intervention_precision": precision,
        "intervention_precision_ci95": list(precision_ci),
        "coverage": coverage,
        "coverage_ci95": list(coverage_ci),
        "population_benefit_rate": pbr,
        "mean_network_ttt_gain": float(gains.mean()) if len(gains) else 0.0,
        "failure_avoidance_rate": len(abstained_failures) / max(1, len(adaptive_failures)),
        "missed_opportunity_rate": len(missed) / max(1, len(wins)),
        "regret_violation_count": sum(bool(row["regret_violation"]) for row in interventions),
        "safety_violation_count": sum(bool(row["safety_violation"]) for row in interventions),
        "legal_violation_count": sum(bool(row["legal_violation"]) for row in interventions),
    }
