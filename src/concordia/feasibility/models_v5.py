from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from concordia.feasibility.hierarchical import HierarchicalSuccessModel
from concordia.feasibility.models import FeasibilityModel


@dataclass
class V5SuccessModel:
    name: str
    kind: str
    global_model: FeasibilityModel | None = None
    hierarchical_model: HierarchicalSuccessModel | None = None
    regime_models: dict[str, FeasibilityModel] | None = None
    mixture_models: tuple["V5SuccessModel", ...] = ()

    @classmethod
    def fit(
        cls,
        name: str,
        matrix: np.ndarray,
        labels: np.ndarray,
        regimes: Sequence[str],
        feature_names: Sequence[str],
        *,
        pooling_strength: float = 30.0,
    ) -> "V5SuccessModel":
        x = np.asarray(matrix, dtype=float)
        y = np.asarray(labels, dtype=int)
        schema = tuple(feature_names)
        if name == "M1_global_logistic":
            indices = tuple(range(max(1, len(schema) - 4)))
            model = FeasibilityModel(
                name,
                "logistic",
                schema,
                feature_indices=indices,
                regularization=0.04,
                iterations=1800,
            ).fit(x, y)
            return cls(name, "global", global_model=model)
        if name == "M2_interaction_logistic":
            model = FeasibilityModel(
                name, "logistic", schema, regularization=0.04, iterations=2000
            ).fit(x, y)
            return cls(name, "global", global_model=model)
        if name == "M3_regime_specific":
            global_model = FeasibilityModel(
                f"{name}_global",
                "logistic",
                schema,
                regularization=0.06,
                iterations=1800,
            ).fit(x, y)
            values = np.asarray(regimes, dtype=object)
            fitted = {}
            for regime in sorted(set(regimes)):
                mask = values == regime
                if int(mask.sum()) < 30 or len(np.unique(y[mask])) < 2:
                    continue
                fitted[regime] = FeasibilityModel(
                    f"{name}_{regime}",
                    "logistic",
                    schema,
                    regularization=0.10,
                    iterations=1800,
                ).fit(x[mask], y[mask])
            return cls(name, "regime_specific", global_model, regime_models=fitted)
        if name == "M4_hierarchical":
            model = HierarchicalSuccessModel.fit(
                x,
                y,
                regimes,
                schema,
                pooling_strength=pooling_strength,
                name=name,
            )
            return cls(name, "hierarchical", hierarchical_model=model)
        if name == "M5_gradient_boosting":
            model = FeasibilityModel(
                name,
                "boosting",
                schema,
                regularization=0.02,
                iterations=100,
                learning_rate=0.06,
            ).fit(x, y)
            return cls(name, "global", global_model=model)
        if name == "M6_mixture":
            hierarchical = cls.fit(
                "M4_hierarchical",
                x,
                y,
                regimes,
                schema,
                pooling_strength=pooling_strength,
            )
            boosting = cls.fit("M5_gradient_boosting", x, y, regimes, schema)
            return cls(name, "mixture", mixture_models=(hierarchical, boosting))
        raise ValueError(f"unknown v5 model: {name}")

    def predict_proba(self, matrix: np.ndarray, regimes: Sequence[str]) -> np.ndarray:
        if self.kind == "global" and self.global_model is not None:
            return self.global_model.predict_proba(matrix)
        if self.kind == "hierarchical" and self.hierarchical_model is not None:
            return self.hierarchical_model.predict_proba(matrix, regimes)
        if self.kind == "regime_specific" and self.global_model is not None:
            output = self.global_model.predict_proba(matrix)
            values = np.asarray(regimes, dtype=object)
            for regime, model in (self.regime_models or {}).items():
                mask = values == regime
                output[mask] = model.predict_proba(matrix[mask])
            return output
        if self.kind == "mixture":
            return np.mean(
                [model.predict_proba(matrix, regimes) for model in self.mixture_models],
                axis=0,
            )
        raise ValueError("v5 success model is incomplete")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "global_model": self.global_model.to_dict() if self.global_model else None,
            "hierarchical_model": self.hierarchical_model.to_dict()
            if self.hierarchical_model
            else None,
            "regime_models": {
                regime: model.to_dict()
                for regime, model in (self.regime_models or {}).items()
            },
            "mixture_models": [model.to_dict() for model in self.mixture_models],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "V5SuccessModel":
        return cls(
            str(value["name"]),
            str(value["kind"]),
            FeasibilityModel.from_dict(value["global_model"])
            if value.get("global_model")
            else None,
            HierarchicalSuccessModel.from_dict(value["hierarchical_model"])
            if value.get("hierarchical_model")
            else None,
            {
                str(regime): FeasibilityModel.from_dict(model)
                for regime, model in value.get("regime_models", {}).items()
            },
            tuple(cls.from_dict(model) for model in value.get("mixture_models", [])),
        )
