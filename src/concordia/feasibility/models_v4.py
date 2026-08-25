from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from concordia.feasibility.models import FeasibilityModel


@dataclass
class V4ProbabilityModel:
    name: str
    feature_names: tuple[str, ...]
    members: list[FeasibilityModel]

    def clone(self, seed: int) -> "V4ProbabilityModel":
        return V4ProbabilityModel(
            self.name,
            self.feature_names,
            [member.clone(seed + index) for index, member in enumerate(self.members)],
        )

    def fit(self, matrix: np.ndarray, labels: np.ndarray) -> "V4ProbabilityModel":
        for member in self.members:
            member.fit(matrix, labels)
        return self

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        return np.mean([member.predict_proba(matrix) for member in self.members], axis=0)

    def importance(self) -> dict[str, float]:
        output = {name: 0.0 for name in self.feature_names}
        for member in self.members:
            for name, value in member.importance().items():
                output[name] = output.get(name, 0.0) + value / len(self.members)
        return output

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "feature_names": list(self.feature_names),
            "members": [member.to_dict() for member in self.members],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "V4ProbabilityModel":
        return cls(
            str(value["name"]),
            tuple(value["feature_names"]),
            [FeasibilityModel.from_dict(member) for member in value["members"]],
        )


@dataclass
class V4BootstrapEnsemble:
    models: list[V4ProbabilityModel]

    @classmethod
    def fit(
        cls,
        prototype: V4ProbabilityModel,
        matrix: np.ndarray,
        labels: np.ndarray,
        groups: Sequence[str],
        *,
        ensemble_size: int,
        seed: int,
    ) -> "V4BootstrapEnsemble":
        rng = np.random.default_rng(seed)
        group_values = np.asarray(groups)
        unique = np.unique(group_values)
        models = []
        attempts = 0
        while len(models) < ensemble_size and attempts < ensemble_size * 20:
            attempts += 1
            selected_groups = rng.choice(unique, len(unique), replace=True)
            indices = np.concatenate(
                [np.flatnonzero(group_values == group) for group in selected_groups]
            )
            if len(np.unique(labels[indices])) < 2:
                continue
            models.append(prototype.clone(seed + attempts * 10).fit(matrix[indices], labels[indices]))
        if not models:
            models.append(prototype.clone(seed).fit(matrix, labels))
        return cls(models)

    def predict(self, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        predictions = np.asarray([model.predict_proba(matrix) for model in self.models])
        return predictions.mean(axis=0), predictions.std(axis=0), np.quantile(predictions, 0.10, axis=0)

    def to_dict(self) -> dict[str, Any]:
        return {"models": [model.to_dict() for model in self.models]}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "V4BootstrapEnsemble":
        return cls([V4ProbabilityModel.from_dict(model) for model in value["models"]])


def build_v4_candidate_models(
    feature_names: Sequence[str], seed: int
) -> list[V4ProbabilityModel]:
    names = tuple(feature_names)
    aps = names.index("alignment_potential_score")

    def logistic(name: str, offset: int, *, features: tuple[int, ...] = (), penalty: float = 0.02):
        return FeasibilityModel(
            name,
            "logistic",
            names,
            feature_indices=features,
            regularization=penalty,
            seed=seed + offset,
        )

    def boosting(name: str, offset: int):
        return FeasibilityModel(
            name,
            "boosting",
            names,
            iterations=35,
            learning_rate=0.08,
            seed=seed + offset,
        )

    return [
        V4ProbabilityModel("M0_APS", names, [logistic("M0_APS", 0, features=(aps,), penalty=0.0)]),
        V4ProbabilityModel("M1_logistic", names, [logistic("M1_logistic", 1)]),
        V4ProbabilityModel(
            "M2_regularized_interactions",
            names,
            [logistic("M2_regularized_interactions", 2, penalty=0.05)],
        ),
        V4ProbabilityModel("M3_gradient_boosting", names, [boosting("M3_gradient_boosting", 3)]),
        V4ProbabilityModel(
            "M4_random_forest",
            names,
            [FeasibilityModel("M4_random_forest", "forest", names, seed=seed + 4)],
        ),
        V4ProbabilityModel("M5_calibrated_gradient_boosting", names, [boosting("M5_base", 5)]),
        V4ProbabilityModel(
            "M6_logistic_gbm_stack",
            names,
            [logistic("M6_logistic", 6), boosting("M6_gbm", 7)],
        ),
    ]
