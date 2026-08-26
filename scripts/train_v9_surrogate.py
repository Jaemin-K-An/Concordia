#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from concordia.v9.action_features import STATE_ACTION_FEATURE_SCHEMA, feature_matrix
from concordia.v9.evaluation import within_state_ranking_metrics
from concordia.v9.surrogate import StateActionTrafficModel


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "artifacts/studies/v9_actionability/raw_metrics.json"
OUTPUT = ROOT / "artifacts/studies/v9_action_models"


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _rows(rows: list[dict], role: str, *, exhaustive: bool = False) -> list[dict]:
    return [
        row for row in rows
        if row["development_role"] == role
        and row["action_id"] != "B6_ALWAYS_ON_REFERENCE"
        and (not exhaustive or row["exhaustive_oracle"])
    ]


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def _regression_metrics(rows: list[dict], prediction: np.ndarray) -> dict:
    actual = np.asarray([row["outcomes"]["tau_t_relative"] for row in rows], dtype=float)
    error = prediction - actual
    grouped = defaultdict(list)
    for row, predicted in zip(rows, prediction):
        grouped[row["state_id"]].append((float(row["outcomes"]["tau_t_relative"]), float(predicted)))
    correlations = []
    for values in grouped.values():
        observed = np.asarray([value[0] for value in values])
        forecast = np.asarray([value[1] for value in values])
        observed_rank, forecast_rank = _rank(observed), _rank(forecast)
        if observed_rank.std() > 1e-12 and forecast_rank.std() > 1e-12:
            correlations.append(float(np.corrcoef(observed_rank, forecast_rank)[0, 1]))
    return {
        "row_count": len(rows),
        "state_count": len(grouped),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "mean_within_state_spearman": float(np.mean(correlations)) if correlations else 0.0,
        **within_state_ranking_metrics(rows, prediction, top_m=5),
    }


def _actionability_diagnostics(rows: list[dict]) -> None:
    experimental = [row for row in rows if row["action_id"] not in {"A00_NULL_B1", "B6_ALWAYS_ON_REFERENCE"}]
    intensity = defaultdict(list)
    pareto = defaultdict(lambda: {"benefit": [], "safety": [], "success": []})
    for row in experimental:
        fraction = f"{float(row['action']['reroute_fraction']):.2f}"
        intensity[fraction].append(float(row["outcomes"]["tau_t_relative"]))
        key = row["action_id"]
        pareto[key]["benefit"].append(float(row["outcomes"]["tau_t_relative"]))
        pareto[key]["safety"].append(float(row["outcomes"]["tau_s"]))
        pareto[key]["success"].append(bool(row["outcomes"]["safe_micro_success"]))
    _write(OUTPUT / "intensity_response.json", {
        key: {
            "evaluated_count": len(values),
            "mean_traffic_benefit": float(np.mean(values)),
            "median_traffic_benefit": float(np.median(values)),
            "positive_rate": float(np.mean(np.asarray(values) > 0.005)),
        }
        for key, values in sorted(intensity.items())
    })
    _write(OUTPUT / "efficiency_safety_pareto.json", {
        key: {
            "mean_traffic_benefit": float(np.mean(value["benefit"])),
            "mean_safety_delta": float(np.mean(value["safety"])),
            "safe_beneficial_rate": float(np.mean(value["success"])),
            "evaluated_count": len(value["benefit"]),
        }
        for key, value in sorted(pareto.items())
    })


def run() -> Path:
    rows = json.loads(DATASET.read_text())
    train = _rows(rows, "train")
    calibration = _rows(rows, "calibration", exhaustive=True)
    if not train or not calibration:
        raise RuntimeError("v9 surrogate requires non-empty train and exhaustive calibration rows")
    x_train = feature_matrix(train)
    y_train = np.asarray([row["outcomes"]["tau_t_relative"] for row in train], dtype=float)
    x_calibration = feature_matrix(calibration)
    candidates = []
    models = []
    for index, (model_id, kind) in enumerate((
        ("T0_random_forest", "random_forest"),
        ("T1_gradient_boosting", "gradient_boosting"),
        ("T2_hist_gradient_boosting", "hist_gradient_boosting"),
    )):
        model = StateActionTrafficModel(
            model_id, kind, tuple(STATE_ACTION_FEATURE_SCHEMA), 9100 + index
        ).fit(x_train, y_train)
        prediction = model.predict(x_calibration)
        metrics = _regression_metrics(calibration, prediction)
        candidates.append({
            "model_id": model_id,
            "kind": kind,
            "training_row_count": len(train),
            "calibration": metrics,
            "feature_importance": model.importance(),
        })
        models.append(model)
        _write(OUTPUT / f"{model_id}.json", model.to_dict())
    selected_index = max(
        range(len(candidates)),
        key=lambda index: (
            candidates[index]["calibration"]["top_5_oracle_recall"],
            candidates[index]["calibration"]["ndcg_action"],
            -candidates[index]["calibration"]["rmse"],
        ),
    )
    selected = models[selected_index]
    summary = {
        "phase": "development_train_and_calibration_only",
        "validation_rows_inspected": 0,
        "candidate_models": candidates,
        "selected_model_id": selected.model_id,
        "selected_on": ["calibration_top_5_oracle_recall", "calibration_ndcg", "calibration_rmse"],
        "gate_B_threshold": 0.80,
        "calibration_gate_B_diagnostic": candidates[selected_index]["calibration"]["top_5_oracle_recall"],
        "validation_gate_B_pending": True,
    }
    _write(OUTPUT / "model_comparison.json", summary)
    _write(OUTPUT / "selected_traffic_model.json", selected.to_dict())
    _write(OUTPUT / "feature_schema.json", {
        "features": list(STATE_ACTION_FEATURE_SCHEMA),
        "feature_count": len(STATE_ACTION_FEATURE_SCHEMA),
        "predecision_only": True,
    })
    _actionability_diagnostics(rows)
    print(OUTPUT / "selected_traffic_model.json")
    return OUTPUT / "selected_traffic_model.json"


if __name__ == "__main__":
    run()
