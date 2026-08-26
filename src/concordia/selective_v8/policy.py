from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from concordia.uplift_v7.learners import RegressionModel
from concordia.uplift_v7.paired_dataset import feature_matrix as v7_feature_matrix

from .safety_filter import CalibratedSafetyFilter
from .traffic_ranker import TrafficRanker


@dataclass
class SafetyFilteredUpliftPolicy:
    traffic_ranker: TrafficRanker
    safety_filter: CalibratedSafetyFilter
    regret_model: RegressionModel
    traffic_rank_percentile_cutoff: float
    regret_threshold: float = 0.08
    policy_name: str = "V8F"

    def predictions(self, rows: Sequence[Mapping[str, object]]) -> dict[str, np.ndarray]:
        traffic = self.traffic_ranker.predict(rows)
        percentile = self.traffic_ranker.percentiles(traffic)
        unsafe = self.safety_filter.probabilities(rows)
        regret = self.regret_model.predict(v7_feature_matrix(rows, self.regret_model.feature_names))
        return {
            "traffic_uplift_score": traffic,
            "traffic_rank_percentile": percentile,
            "unsafe_probability": unsafe,
            "predicted_regret": regret,
        }

    def decide(self, rows: Sequence[Mapping[str, object]]) -> list[dict[str, Any]]:
        prediction = self.predictions(rows)
        output = []
        for index, row in enumerate(rows):
            traffic_pass = prediction["traffic_rank_percentile"][index] >= self.traffic_rank_percentile_cutoff
            safety_pass = prediction["unsafe_probability"][index] <= self.safety_filter.unsafe_probability_threshold
            regret_pass = prediction["predicted_regret"][index] <= self.regret_threshold
            legal = bool(row.get("legal_executable_predecision", True))
            intervene = bool(traffic_pass and safety_pass and regret_pass and legal)
            if not traffic_pass:
                reason = "traffic_rank_below_cutoff"
            elif not safety_pass:
                reason = "unsafe_probability_veto"
            elif not regret_pass:
                reason = "regret_veto"
            elif not legal:
                reason = "illegal_or_unexecutable"
            else:
                reason = "intervene"
            output.append(
                {
                    "pair_id": str(row["pair_id"]),
                    "intervene": intervene,
                    "executed_policy": "Adaptive" if intervene else "B1",
                    "reason": reason,
                    **{key: float(value[index]) for key, value in prediction.items()},
                }
            )
        return output

    def to_dict(self) -> dict[str, Any]:
        return {
            "traffic_ranker": self.traffic_ranker.to_dict(),
            "safety_filter": self.safety_filter.to_dict(),
            "regret_model": self.regret_model.to_dict(),
            "traffic_rank_percentile_cutoff": self.traffic_rank_percentile_cutoff,
            "regret_threshold": self.regret_threshold,
            "policy_name": self.policy_name,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SafetyFilteredUpliftPolicy":
        return cls(
            TrafficRanker.from_dict(value["traffic_ranker"]),
            CalibratedSafetyFilter.from_dict(value["safety_filter"]),
            RegressionModel.from_dict(value["regret_model"]),
            float(value["traffic_rank_percentile_cutoff"]),
            float(value.get("regret_threshold", 0.08)),
            str(value.get("policy_name", "V8F")),
        )

