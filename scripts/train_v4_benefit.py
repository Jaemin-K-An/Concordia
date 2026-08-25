#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from concordia.feasibility import BenefitModel, V4_FEATURE_SCHEMA, regression_metrics


ROOT = Path(__file__).resolve().parents[1]
MODEL_STUDY = ROOT / "artifacts/studies/v4_model_selection"
STUDY = ROOT / "artifacts/studies/v4_precision_validation"


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _matrix(rows: list[dict]) -> np.ndarray:
    return np.asarray(
        [[row["features"][name] for name in V4_FEATURE_SCHEMA] for row in rows],
        dtype=float,
    )


def _lower_from_mean(model: BenefitModel, z: float) -> BenefitModel:
    value = BenefitModel.from_dict(model.to_dict())
    shift = z * value.parameters["residual_standard_deviation"]
    if value.kind in {"ridge", "elastic"}:
        value.parameters["intercept"] -= shift
    else:
        value.parameters["base"] -= shift
    value.name = f"{model.name}_lcb"
    return value


def run() -> Path:
    rows = json.loads((MODEL_STUDY / "raw_metrics.json").read_text(encoding="utf-8"))
    train = [row for row in rows if row["development_role"] == "training"]
    evaluation = [
        row for row in rows if row["development_role"] == "calibration_evaluation"
    ]
    train_x = _matrix(train)
    train_y = np.asarray([row["targets"]["relative_ttt_gain"] for row in train])
    evaluation_x = _matrix(evaluation)
    evaluation_y = np.asarray(
        [row["targets"]["relative_ttt_gain"] for row in evaluation]
    )
    mean_models = [
        BenefitModel("ridge", "ridge", penalty=1e-3),
        BenefitModel("elastic_net", "elastic", penalty=2e-3),
        BenefitModel("gradient_boosting", "boosting", iterations=140),
    ]
    mean_results = {}
    for model in mean_models:
        model.fit(train_x, train_y)
        mean_results[model.name] = regression_metrics(
            evaluation_y, model.predict(evaluation_x)
        )
    selected_mean = min(mean_models, key=lambda model: mean_results[model.name]["mae"])
    lower_models = [_lower_from_mean(model, 1.282) for model in mean_models]
    lower_models.append(
        BenefitModel(
            "quantile_gradient_boosting_q10", "boosting", quantile=0.10, iterations=180
        ).fit(train_x, train_y)
    )
    lower_results = {
        model.name: regression_metrics(evaluation_y, model.predict(evaluation_x), quantile=0.10)
        for model in lower_models
    }
    reliable = [
        model for model in lower_models if lower_results[model.name]["lower_bound_coverage"] >= 0.85
    ]
    selected_lower = min(
        reliable or lower_models,
        key=lambda model: (
            lower_results[model.name]["pinball_loss"],
            -lower_results[model.name]["lower_bound_coverage"],
        ),
    )
    final_rows = [row for row in rows if row["development_role"] != "validation"]
    final_x = _matrix(final_rows)
    final_y = np.asarray([row["targets"]["relative_ttt_gain"] for row in final_rows])
    final_mean = BenefitModel(
        selected_mean.name,
        selected_mean.kind,
        quantile=selected_mean.quantile,
        penalty=selected_mean.penalty,
        iterations=selected_mean.iterations,
        learning_rate=selected_mean.learning_rate,
    ).fit(final_x, final_y)
    if selected_lower.name.endswith("_lcb"):
        base_name = selected_lower.name.removesuffix("_lcb")
        base = next(model for model in mean_models if model.name == base_name)
        refit = BenefitModel(
            base.name,
            base.kind,
            quantile=base.quantile,
            penalty=base.penalty,
            iterations=base.iterations,
            learning_rate=base.learning_rate,
        ).fit(final_x, final_y)
        final_lower = _lower_from_mean(refit, 1.282)
    else:
        final_lower = BenefitModel(
            selected_lower.name,
            selected_lower.kind,
            quantile=selected_lower.quantile,
            penalty=selected_lower.penalty,
            iterations=selected_lower.iterations,
            learning_rate=selected_lower.learning_rate,
        ).fit(final_x, final_y)
    package = {
        "complete": True,
        "mean_model": final_mean.to_dict(),
        "lower_model": final_lower.to_dict(),
        "mean_model_selection": mean_results,
        "lower_model_selection": lower_results,
        "selected_mean_model": selected_mean.name,
        "selected_lower_model": selected_lower.name,
        "training_case_ids": [row["case_id"] for row in final_rows],
        "validation_case_ids": [],
        "final_holdout_case_ids": [],
    }
    output = STUDY / "benefit_package.json"
    _write(output, package)
    _write(
        STUDY / "benefit_summary.json",
        {
            "complete": True,
            "selected_mean_model": selected_mean.name,
            "selected_lower_model": selected_lower.name,
            "evaluation_mean_metrics": mean_results[selected_mean.name],
            "evaluation_lower_metrics": lower_results[selected_lower.name],
            "final_holdout_used": False,
        },
    )
    print(output)
    return output


if __name__ == "__main__":
    run()
