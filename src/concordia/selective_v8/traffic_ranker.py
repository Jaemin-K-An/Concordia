from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from concordia.safety_v8.candidate_conditioned import empirical_rank_percentile
from concordia.uplift_v7.paired_dataset import feature_matrix
from concordia.uplift_v7.treatment_effect import CausalEffectLearner


@dataclass
class TrafficRanker:
    model: CausalEffectLearner
    development_score_reference: tuple[float, ...]
    provenance: str = "v7_C0_direct_paired_random_forest"

    def predict(self, rows: Sequence[Mapping[str, object]]) -> np.ndarray:
        return self.model.predict(feature_matrix(rows, self.model.feature_names))

    def percentiles(self, scores: Sequence[float]) -> np.ndarray:
        return empirical_rank_percentile(scores, self.development_score_reference)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.to_dict(),
            "development_score_reference": list(self.development_score_reference),
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrafficRanker":
        return cls(
            CausalEffectLearner.from_dict(value["model"]),
            tuple(float(item) for item in value["development_score_reference"]),
            str(value.get("provenance", "unknown")),
        )

