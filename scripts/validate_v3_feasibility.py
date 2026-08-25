#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/concordia-matplotlib-v3")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/concordia-cache-v3")

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from concordia.feasibility import (
    BootstrapFeasibilityEnsemble,
    FEATURE_SCHEMA,
    classification_metrics,
    load_model,
    select_intervention_threshold,
)


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "artifacts" / "studies" / "v3_feasibility_prediction"


def _write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run() -> Path:
    prereg = yaml.safe_load((ROOT / "configs/v3/preregistration.yaml").read_text())
    config = yaml.safe_load((ROOT / "configs/v3/model_selection.yaml").read_text())
    rows = json.loads((STUDY / "raw_metrics.json").read_text(encoding="utf-8"))
    trained = json.loads((STUDY / "trained_candidates.json").read_text(encoding="utf-8"))
    train = [row for row in rows if row["source_split"] == "training"]
    validation = [row for row in rows if row["source_split"] == "validation"]

    def matrix(selected):
        return np.asarray(
            [[row["features"][name] for name in FEATURE_SCHEMA] for row in selected],
            dtype=float,
        )

    train_x = matrix(train)
    validation_x = matrix(validation)
    validation_y = np.asarray([int(row["label"] == "WIN") for row in validation])
    evaluations = {}
    models = []
    for value in trained["candidate_models"]:
        model = load_model(value)
        models.append(model)
        evaluations[model.name] = classification_metrics(
            validation_y, model.predict_proba(validation_x)
        )
    selected = max(
        models,
        key=lambda model: (
            evaluations[model.name]["pr_auc"] or -1.0,
            -evaluations[model.name]["brier_score"],
            -evaluations[model.name]["ece"],
            model.kind == "logistic",
        ),
    )
    groups = [f"{row['scenario']}-s{row['seed']}" for row in train]
    train_y = np.asarray([int(row["label"] == "WIN") for row in train])
    ensemble = BootstrapFeasibilityEnsemble.fit(
        selected,
        train_x,
        train_y,
        groups,
        ensemble_size=int(config["bootstrap_ensemble_size"]),
        seed=int(config["bootstrap_seed"]),
    )
    probabilities, uncertainty, lower = ensemble.predict(validation_x)
    threshold = select_intervention_threshold(
        validation_y,
        probabilities,
        uncertainty,
        precision_target=float(prereg["primary"]["intervention_precision_target"]),
        coverage_target=float(prereg["primary"]["coverage_target"]),
        maximum_uncertainty=float(config["maximum_uncertainty"]),
        candidates=config["threshold_candidates"],
    )
    aps_values = np.asarray(
        [row["features"]["alignment_potential_score"] for row in validation]
    )
    aps_rows = []
    for aps_threshold in config["aps_threshold_candidates"]:
        intervene = aps_values >= float(aps_threshold)
        aps_rows.append(
            {
                "threshold": float(aps_threshold),
                "precision": float(validation_y[intervene].mean()) if intervene.any() else 0.0,
                "coverage": float(intervene.mean()),
            }
        )
    aps_selected = max(
        aps_rows,
        key=lambda row: (
            row["precision"] >= prereg["primary"]["intervention_precision_target"],
            row["coverage"] if row["precision"] >= prereg["primary"]["intervention_precision_target"] else row["precision"],
        ),
    )
    selected_metrics = classification_metrics(
        validation_y, probabilities, threshold["selected"]["threshold"]
    )
    feature_importance = sorted(
        selected.importance().items(), key=lambda item: item[1], reverse=True
    )
    package = {
        "complete": True,
        "selected_model_name": selected.name,
        "selected_model_kind": selected.kind,
        "ensemble": ensemble.to_dict(),
        "benefit_regression": trained["benefit_regression"],
        "safety_regression": trained["safety_regression"],
        "threshold_selection": threshold,
        "aps_threshold_selection": {"selected": aps_selected, "curve": aps_rows},
        "maximum_uncertainty": float(config["maximum_uncertainty"]),
        "minimum_acceptance_probability": float(config["minimum_acceptance_probability"]),
        "maximum_tail_loss": float(config["maximum_tail_loss"]),
        "benefit_lcb_z": float(config["benefit_lcb_z"]),
        "training_case_ids": trained["training_case_ids"],
        "validation_case_ids": [row["case_id"] for row in validation],
        "holdout_case_ids": [],
        "holdout_used": False,
        "feature_names": list(FEATURE_SCHEMA),
        "feature_importance": feature_importance,
        "split_hash": trained["split_hash"],
        "feature_schema_hash": trained["feature_schema_hash"],
    }
    _write(STUDY / "selected_model.json", package)
    statistics = {
        "candidate_model_validation": evaluations,
        "selected_ensemble_validation": selected_metrics,
        "validation_label_counts": dict(Counter(row["label"] for row in validation)),
        "secondary_three_class_available": bool(trained["secondary_multiclass_models"]),
    }
    _write(STUDY / "statistical_tests.json", statistics)
    _write(STUDY / "processed_metrics.json", package)
    summary = {
        "complete": True,
        "study": "Study V — Alignment Feasibility Prediction",
        "development_validation_only": True,
        "selected_model": selected.name,
        "validation_metrics": selected_metrics,
        "selected_threshold_candidate": threshold["selected"],
        "selected_aps_threshold_candidate": aps_selected,
        "feature_importance_top10": feature_importance[:10],
        "holdout_used_for_selection": False,
        "claim_boundary": "Study V is not primary success evidence.",
    }
    _write(STUDY / "summary.json", summary)
    figure_dir = STUDY / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(5.4, 5.0))
    curve = selected_metrics["calibration_curve"]
    axis.plot(
        [item["mean_predicted_probability"] for item in curve],
        [item["observed_win_rate"] for item in curve],
        marker="o",
    )
    axis.plot([0, 1], [0, 1], linestyle="--", color="#888888")
    axis.set(xlabel="Predicted P(WIN)", ylabel="Observed WIN rate", title="Validation calibration")
    fig.tight_layout()
    fig.savefig(figure_dir / "calibration_curve.png", dpi=180)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(6.2, 4.2))
    risk_curve = threshold["risk_coverage_curve"]
    axis.plot([row["coverage"] for row in risk_curve], [row["precision"] for row in risk_curve], marker="o")
    axis.axhline(prereg["primary"]["intervention_precision_target"], linestyle="--", color="#777777")
    axis.axvline(prereg["primary"]["coverage_target"], linestyle=":", color="#777777")
    axis.set(xlabel="Coverage", ylabel="Intervention precision", title="Validation risk–coverage")
    fig.tight_layout()
    fig.savefig(figure_dir / "validation_risk_coverage.png", dpi=180)
    plt.close(fig)
    print(STUDY / "summary.json")
    return STUDY / "summary.json"


if __name__ == "__main__":
    run()
