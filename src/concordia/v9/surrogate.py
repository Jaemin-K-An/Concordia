from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from concordia.uplift_v7.learners import RegressionModel, build_regression_model


@dataclass
class StateActionTrafficModel:
    """Serializable state-action regressor with optional histogram quantization."""

    model_id: str
    kind: str
    feature_names: tuple[str, ...]
    seed: int
    base_model: RegressionModel | None = None
    bin_edges: tuple[tuple[float, ...], ...] | None = None

    def _quantize(self, matrix: np.ndarray, *, fit: bool) -> np.ndarray:
        values = np.asarray(matrix, dtype=float)
        if self.kind != "hist_gradient_boosting":
            return values
        if fit:
            self.bin_edges = tuple(
                tuple(sorted(set(map(float, np.quantile(values[:, index], np.linspace(0.0, 1.0, 17))[1:-1]))))
                for index in range(values.shape[1])
            )
        if self.bin_edges is None:
            raise ValueError("histogram traffic model has no fitted bin edges")
        return np.column_stack([
            np.searchsorted(np.asarray(edges), values[:, index], side="right")
            for index, edges in enumerate(self.bin_edges)
        ]).astype(float)

    def fit(self, matrix: np.ndarray, target: Sequence[float]) -> "StateActionTrafficModel":
        transformed = self._quantize(np.asarray(matrix, dtype=float), fit=True)
        base_kind = "gradient_boosting" if self.kind == "hist_gradient_boosting" else self.kind
        self.base_model = build_regression_model(
            base_kind, self.feature_names, self.seed, self.model_id
        ).fit(transformed, np.asarray(target, dtype=float))
        return self

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        if self.base_model is None:
            raise ValueError("state-action traffic model is not fitted")
        return self.base_model.predict(self._quantize(np.asarray(matrix, dtype=float), fit=False))

    def importance(self) -> dict[str, float]:
        return self.base_model.importance() if self.base_model else {name: 0.0 for name in self.feature_names}

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "kind": self.kind,
            "feature_names": list(self.feature_names),
            "seed": self.seed,
            "base_model": self.base_model.to_dict() if self.base_model else None,
            "bin_edges": [list(edges) for edges in self.bin_edges] if self.bin_edges else None,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StateActionTrafficModel":
        model = cls(
            str(value["model_id"]), str(value["kind"]),
            tuple(value["feature_names"]), int(value["seed"]),
        )
        if value.get("base_model") is not None:
            model.base_model = RegressionModel.from_dict(value["base_model"])
        if value.get("bin_edges") is not None:
            model.bin_edges = tuple(tuple(map(float, edges)) for edges in value["bin_edges"])
        return model
