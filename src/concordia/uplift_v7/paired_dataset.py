from __future__ import annotations

import math
from typing import Mapping, Sequence

from concordia.errors import ValidationError
from concordia.micro_v6.features import MICRO_V6_FEATURE_GROUPS, MICRO_V6_FEATURE_SCHEMA

from .outcomes import paired_treatment_outcomes


UPLIFT_V7_ADDITIONAL_FEATURES = (
    "edge_disjointness",
    "number_route_alternatives",
    "preference_entropy_proxy",
    "topology_asymmetric",
    "v6_micro_success_score",
)
UPLIFT_V7_FEATURE_SCHEMA = (*MICRO_V6_FEATURE_SCHEMA, *UPLIFT_V7_ADDITIONAL_FEATURES)
UPLIFT_V7_FEATURE_GROUPS = {
    **MICRO_V6_FEATURE_GROUPS,
    "topology_v7": ("edge_disjointness", "number_route_alternatives", "topology_asymmetric"),
    "preference_v7": ("preference_entropy_proxy",),
    "v6_score": ("v6_micro_success_score",),
}

FORBIDDEN_FEATURE_TOKENS = (
    "adaptive_ttt",
    "realized",
    "post_treatment",
    "future",
    "tau_t",
    "tau_s",
    "outcome",
)


def enrich_predecision_features(
    features: Mapping[str, object],
    condition: Mapping[str, object],
    *,
    v6_micro_success_score: float,
    number_route_alternatives: int | None = None,
) -> dict[str, float]:
    output = {name: float(features[name]) for name in MICRO_V6_FEATURE_SCHEMA}
    topology = str(condition.get("topology", "unknown"))
    asymmetric = topology == "asymmetric"
    if asymmetric:
        output["topology_real_like"] = 0.0
    overlap = float(output["route_overlap"])
    preference_variance = max(0.0, float(output["preference_variance"]))
    heterogeneity_mass = sum(
        float(output[name])
        for name in (
            "heterogeneity_low",
            "heterogeneity_medium",
            "heterogeneity_high",
            "heterogeneity_bimodal",
            "heterogeneity_long_tail",
        )
    )
    output.update(
        {
            "edge_disjointness": 1.0 - overlap,
            "number_route_alternatives": float(
                number_route_alternatives
                if number_route_alternatives is not None
                else 3 if asymmetric else 2
            ),
            "preference_entropy_proxy": math.log1p(preference_variance)
            * max(heterogeneity_mass, 1.0),
            "topology_asymmetric": float(asymmetric),
            "v6_micro_success_score": float(v6_micro_success_score),
        }
    )
    validate_predecision_features(output)
    return {name: output[name] for name in UPLIFT_V7_FEATURE_SCHEMA}


def validate_predecision_features(features: Mapping[str, object]) -> None:
    if set(features) != set(UPLIFT_V7_FEATURE_SCHEMA):
        raise ValidationError("v7 uplift feature schema mismatch")
    for name, value in features.items():
        lowered = name.lower()
        if any(token in lowered for token in FORBIDDEN_FEATURE_TOKENS):
            raise ValidationError(f"v7 post-treatment leakage in feature: {name}")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValidationError(f"non-finite v7 feature: {name}")


def paired_row_from_v6(
    row: Mapping[str, object],
    *,
    v6_micro_success_score: float,
    source: str,
) -> dict:
    baseline = dict(row["counterfactual_B1"])
    adaptive = dict(row["counterfactual_adaptive"])
    outcomes = paired_treatment_outcomes(baseline, adaptive)
    condition = dict(row["condition"])
    features = enrich_predecision_features(
        row["features_pre_decision"],
        condition,
        v6_micro_success_score=v6_micro_success_score,
    )
    pair_id = f"v7-historical::{row['case_id']}"
    return {
        "pair_id": pair_id,
        "source_case_id": row["case_id"],
        "source": source,
        "seed": int(row["seed"]),
        "condition": condition,
        "decision_time": float(row["decision_time"]),
        "feature_observation_end_time": float(row["feature_observation_end_time"]),
        "predecision_features": features,
        "ttt_b1": float(baseline["total_travel_time_seconds"]),
        "ttt_adaptive": float(adaptive["total_travel_time_seconds"]),
        "risk_b1": float(baseline["safety"]["cvar_drac_95"]),
        "risk_adaptive": float(adaptive["safety"]["cvar_drac_95"]),
        "generated_vehicle_count": int(baseline["generated_vehicle_count"]),
        "outcomes": outcomes.to_dict(),
        "counterfactual_B1": baseline,
        "counterfactual_adaptive": adaptive,
        "pairing": {
            **dict(row["pairing"]),
            "metadata_identical_except_treatment": bool(all(row["pairing"].values())),
            "treatment_only_difference": True,
        },
    }


def feature_matrix(
    rows: Sequence[Mapping[str, object]],
    feature_names: Sequence[str] = UPLIFT_V7_FEATURE_SCHEMA,
):
    import numpy as np

    return np.asarray(
        [[float(row["predecision_features"][name]) for name in feature_names] for row in rows],
        dtype=float,
    )

