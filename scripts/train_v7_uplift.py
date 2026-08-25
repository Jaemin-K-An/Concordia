#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from concordia.micro_v6.features import MICRO_V6_FEATURE_SCHEMA
from concordia.uplift_v7.conformal import ConformalAdjustments
from concordia.uplift_v7.evaluation import (
    cumulative_gain,
    deployment_metrics,
    effect_calibration,
    regression_metrics,
)
from concordia.uplift_v7.learners import build_regression_model
from concordia.uplift_v7.paired_dataset import (
    UPLIFT_V7_FEATURE_GROUPS,
    UPLIFT_V7_FEATURE_SCHEMA,
    feature_matrix,
)
from concordia.uplift_v7.policy import UpliftPolicy
from concordia.uplift_v7.quantiles import (
    BootstrapCausalEnsemble,
    BootstrapRegressionEnsemble,
    PinballQuantileRegressor,
)
from concordia.uplift_v7.treatment_effect import CausalEffectLearner
from v6_frozen import load_policy as load_v6_policy


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "artifacts/studies/v7_paired_dataset/raw_metrics.json"
MODEL_OUTPUT = ROOT / "artifacts/studies/v7_model_selection"
POLICY_OUTPUT = ROOT / "artifacts/studies/v7_policy_validation"


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _arrays(rows: list[dict], feature_names=UPLIFT_V7_FEATURE_SCHEMA) -> dict[str, np.ndarray]:
    return {
        "matrix": feature_matrix(rows, feature_names),
        "traffic": np.asarray(
            [row["outcomes"]["tau_t_relative"] for row in rows], dtype=float
        ),
        "safety": np.asarray([row["outcomes"]["tau_s"] for row in rows], dtype=float),
        "regret": np.asarray(
            [row["outcomes"]["max_regret"] for row in rows], dtype=float
        ),
        "ttt_b1": np.asarray([row["ttt_b1"] for row in rows], dtype=float),
        "ttt_adaptive": np.asarray([row["ttt_adaptive"] for row in rows], dtype=float),
        "generated": np.asarray(
            [row["generated_vehicle_count"] for row in rows], dtype=float
        ),
    }


def _fit_traffic(
    formulation: str,
    kind: str,
    values: dict[str, np.ndarray],
    feature_names: tuple[str, ...],
    seed: int,
) -> CausalEffectLearner:
    return CausalEffectLearner.fit(
        formulation,
        kind,
        values["matrix"],
        values["traffic"],
        values["ttt_b1"],
        values["ttt_adaptive"],
        values["generated"],
        feature_names,
        seed=seed,
    )


def _bounds(
    method: str,
    matrix: np.ndarray,
    traffic_model: CausalEffectLearner,
    safety_model,
    regret_model,
    traffic_bootstrap: BootstrapCausalEnsemble,
    safety_bootstrap: BootstrapRegressionEnsemble,
    regret_bootstrap: BootstrapRegressionEnsemble,
    conformal: ConformalAdjustments,
) -> dict[str, np.ndarray]:
    traffic = traffic_model.predict(matrix)
    safety = safety_model.predict(matrix)
    regret = regret_model.predict(matrix)
    if method == "mean":
        return {
            "traffic_mean": traffic,
            "traffic_lower": traffic,
            "safety_mean": safety,
            "safety_upper": safety,
            "regret_mean": regret,
            "regret_upper": regret,
        }
    if method == "bootstrap_quantile":
        _mean, traffic_lower, _upper = traffic_bootstrap.interval(matrix, 0.10, 0.90)
        _mean, _lower, safety_upper = safety_bootstrap.interval(matrix, 0.10, 0.90)
        _mean, _lower, regret_upper = regret_bootstrap.interval(matrix, 0.10, 0.90)
        return {
            "traffic_mean": traffic,
            "traffic_lower": traffic_lower,
            "safety_mean": safety,
            "safety_upper": safety_upper,
            "regret_mean": regret,
            "regret_upper": regret_upper,
        }
    return {
        "traffic_mean": traffic,
        "traffic_lower": traffic - conformal.traffic_radius,
        "safety_mean": safety,
        "safety_upper": safety + conformal.safety_upper_adjustment,
        "regret_mean": regret,
        "regret_upper": regret + conformal.regret_upper_adjustment,
    }


def _mask(rows: list[dict], bounds: dict[str, np.ndarray], thresholds: tuple[float, float, float]):
    traffic, safety, regret = thresholds
    return np.asarray(
        [
            bounds["traffic_lower"][index] > traffic
            and bounds["safety_upper"][index] <= safety
            and bounds["regret_upper"][index] <= regret
            and bool(row["outcomes"]["legal"])
            for index, row in enumerate(rows)
        ],
        dtype=bool,
    )


def _v6_style_metrics(rows: list[dict]) -> dict:
    converted = []
    for row in rows:
        converted.append(
            {
                "case_id": row["pair_id"],
                "features_pre_decision": {
                    name: row["predecision_features"][name]
                    for name in MICRO_V6_FEATURE_SCHEMA
                },
                "label": {"safe_micro_success": row["outcomes"]["safe_micro_success"]},
            }
        )
    decisions = load_v6_policy().decide(converted)
    return deployment_metrics(rows, [row["intervene"] for row in decisions])


def _scenario_metrics(rows: list[dict], selected: np.ndarray) -> list[dict]:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[(str(row["condition"]["topology"]), str(row["condition"]["penetration"]))].append(
            index
        )
    output = []
    for (topology, penetration), indices in sorted(groups.items()):
        subset = [rows[index] for index in indices]
        metrics = deployment_metrics(subset, selected[indices])
        output.append(
            {"topology": topology, "penetration": float(penetration), **metrics}
        )
    return output


def _scenario_family_holdouts(
    rows: list[dict], formulation: str, kind: str, seed: int
) -> list[dict]:
    holdouts = (
        (
            "leave_asymmetric_topology_out",
            lambda row: row["condition"]["topology"] == "asymmetric",
        ),
        (
            "leave_high_demand_band_out",
            lambda row: float(row["condition"]["demand"]) > 1300,
        ),
        (
            "leave_full_penetration_band_out",
            lambda row: float(row["condition"]["penetration"]) == 1.0,
        ),
    )
    output = []
    for index, (name, predicate) in enumerate(holdouts):
        training = [row for row in rows if not predicate(row)]
        testing = [row for row in rows if predicate(row)]
        train_values = _arrays(training)
        test_values = _arrays(testing)
        model = _fit_traffic(
            formulation,
            kind,
            train_values,
            tuple(UPLIFT_V7_FEATURE_SCHEMA),
            seed + 7000 + index,
        )
        output.append(
            {
                "holdout": name,
                "train_pair_count": len(training),
                "test_pair_count": len(testing),
                "metrics": regression_metrics(
                    test_values["traffic"], model.predict(test_values["matrix"])
                ),
                "selection_note": "development-only architecture transferred without threshold retuning",
            }
        )
    return output


def run(*, force: bool = False) -> Path:
    package_path = POLICY_OUTPUT / "selected_policy_package.json"
    summary_path = POLICY_OUTPUT / "validation_summary.json"
    if package_path.is_file() and summary_path.is_file() and not force:
        print(package_path)
        return package_path
    rows = json.loads(DATASET.read_text())
    if len(rows) < 1000:
        raise RuntimeError("v7 uplift training requires at least 1,000 paired cases")
    config = yaml.safe_load((ROOT / "configs/v7/model.yaml").read_text())
    prereg = yaml.safe_load((ROOT / "configs/v7/preregistration.yaml").read_text())
    roles = {
        role: [row for row in rows if row["development_role"] == role]
        for role in ("train", "calibration", "validation")
    }
    values = {role: _arrays(role_rows) for role, role_rows in roles.items()}
    seed = int(config["seed"])
    candidates = []
    fitted: dict[str, CausalEffectLearner] = {}
    offset = 0
    for formulation in config["causal_formulations"]:
        for kind in config["traffic_candidates"]:
            model = _fit_traffic(
                formulation,
                kind,
                values["train"],
                tuple(UPLIFT_V7_FEATURE_SCHEMA),
                seed + offset,
            )
            key = f"{formulation}|{kind}"
            prediction = model.predict(values["validation"]["matrix"])
            metrics = regression_metrics(values["validation"]["traffic"], prediction)
            candidates.append(
                {
                    "key": key,
                    "formulation": formulation,
                    "model": kind,
                    "validation_metrics": metrics,
                    "importance": model.importance(),
                }
            )
            fitted[key] = model
            offset += 1
    selected_traffic_entry = min(
        candidates,
        key=lambda row: (
            row["validation_metrics"]["rmse"],
            row["validation_metrics"]["mae"],
            -row["validation_metrics"]["spearman"],
        ),
    )
    selected_traffic = fitted[selected_traffic_entry["key"]]

    direct_quantiles = {
        "traffic_q10": PinballQuantileRegressor(
            0.10, tuple(UPLIFT_V7_FEATURE_SCHEMA)
        ).fit(values["train"]["matrix"], values["train"]["traffic"]),
        "traffic_q50": PinballQuantileRegressor(
            0.50, tuple(UPLIFT_V7_FEATURE_SCHEMA)
        ).fit(values["train"]["matrix"], values["train"]["traffic"]),
        "safety_q90": PinballQuantileRegressor(
            0.90, tuple(UPLIFT_V7_FEATURE_SCHEMA)
        ).fit(values["train"]["matrix"], values["train"]["safety"]),
        "regret_q90": PinballQuantileRegressor(
            0.90, tuple(UPLIFT_V7_FEATURE_SCHEMA)
        ).fit(values["train"]["matrix"], values["train"]["regret"]),
    }
    traffic_q50_prediction = direct_quantiles["traffic_q50"].predict(
        values["validation"]["matrix"]
    )
    candidates.append(
        {
            "key": "C0_direct_paired|linear_pinball_q50",
            "formulation": "C0_direct_paired",
            "model": "linear_pinball_q50",
            "validation_metrics": regression_metrics(
                values["validation"]["traffic"], traffic_q50_prediction
            ),
            "importance": {},
            "selection_role": "quantile_diagnostic_not_mean-model_candidate",
        }
    )

    auxiliary_comparison = {"safety": [], "regret": []}
    auxiliary_fitted = {"safety": {}, "regret": {}}
    for target_name, candidate_key in (
        ("safety", "safety_candidates"),
        ("regret", "regret_candidates"),
    ):
        for index, kind in enumerate(config[candidate_key]):
            model = build_regression_model(
                kind,
                tuple(UPLIFT_V7_FEATURE_SCHEMA),
                seed + 100 + index + 10 * (target_name == "regret"),
                f"v7_{target_name}",
            ).fit(values["train"]["matrix"], values["train"][target_name])
            metrics = regression_metrics(
                values["validation"][target_name],
                model.predict(values["validation"]["matrix"]),
            )
            auxiliary_comparison[target_name].append(
                {"model": kind, "validation_metrics": metrics, "importance": model.importance()}
            )
            auxiliary_fitted[target_name][kind] = model
    selected_auxiliary = {
        name: min(
            entries,
            key=lambda row: (
                row["validation_metrics"]["rmse"], row["validation_metrics"]["mae"]
            ),
        )["model"]
        for name, entries in auxiliary_comparison.items()
    }
    safety_model = auxiliary_fitted["safety"][selected_auxiliary["safety"]]
    regret_model = auxiliary_fitted["regret"][selected_auxiliary["regret"]]

    member_count = int(config["bootstrap_members"])
    traffic_bootstrap = BootstrapCausalEnsemble.fit(
        selected_traffic.formulation,
        selected_traffic.kind,
        values["train"]["matrix"],
        values["train"]["traffic"],
        values["train"]["ttt_b1"],
        values["train"]["ttt_adaptive"],
        values["train"]["generated"],
        UPLIFT_V7_FEATURE_SCHEMA,
        member_count=member_count,
        seed=seed + 1000,
    )
    safety_bootstrap = BootstrapRegressionEnsemble.fit(
        selected_auxiliary["safety"],
        values["train"]["matrix"],
        values["train"]["safety"],
        UPLIFT_V7_FEATURE_SCHEMA,
        member_count=member_count,
        seed=seed + 2000,
        name="v7_safety_bootstrap",
    )
    regret_bootstrap = BootstrapRegressionEnsemble.fit(
        selected_auxiliary["regret"],
        values["train"]["matrix"],
        values["train"]["regret"],
        UPLIFT_V7_FEATURE_SCHEMA,
        member_count=member_count,
        seed=seed + 3000,
        name="v7_regret_bootstrap",
    )
    calibration_prediction = {
        "traffic": selected_traffic.predict(values["calibration"]["matrix"]),
        "safety": safety_model.predict(values["calibration"]["matrix"]),
        "regret": regret_model.predict(values["calibration"]["matrix"]),
    }
    conformal_by_alpha = {
        str(alpha): ConformalAdjustments.fit(
            float(alpha),
            values["calibration"]["traffic"],
            calibration_prediction["traffic"],
            values["calibration"]["safety"],
            calibration_prediction["safety"],
            values["calibration"]["regret"],
            calibration_prediction["regret"],
        )
        for alpha in config["conformal_miscoverage"]
    }

    frontier = []
    validation_bounds = {}
    default_conformal = conformal_by_alpha[str(config["conformal_miscoverage"][0])]
    methods = [("mean", None), ("bootstrap_quantile", None)] + [
        ("conformalized_residual", str(alpha))
        for alpha in config["conformal_miscoverage"]
    ]
    for method, alpha in methods:
        conformal = conformal_by_alpha[alpha] if alpha else default_conformal
        bounds = _bounds(
            method,
            values["validation"]["matrix"],
            selected_traffic,
            safety_model,
            regret_model,
            traffic_bootstrap,
            safety_bootstrap,
            regret_bootstrap,
            conformal,
        )
        validation_bounds[f"{method}|{alpha}"] = bounds
        for traffic_threshold in config["traffic_lcb_thresholds"]:
            for safety_threshold in config["safety_ucb_thresholds"]:
                for regret_threshold in config["regret_ucb_thresholds"]:
                    thresholds = (
                        float(traffic_threshold),
                        float(safety_threshold),
                        float(regret_threshold),
                    )
                    selected = _mask(roles["validation"], bounds, thresholds)
                    metrics = deployment_metrics(roles["validation"], selected)
                    eligible = bool(
                        method != "mean"
                        and thresholds[0]
                        >= float(prereg["outcomes"]["traffic"]["minimum_relative_uplift"])
                        and metrics["intervention_count"]
                        >= int(config["minimum_validation_interventions"])
                        and metrics["deployment_precision"]
                        >= float(prereg["targets"]["deployment_precision"])
                        and metrics["safety_violation_count"] == 0
                    )
                    frontier.append(
                        {
                            "method": method,
                            "conformal_miscoverage": float(alpha) if alpha else None,
                            "traffic_lcb_threshold": thresholds[0],
                            "safety_ucb_threshold": thresholds[1],
                            "regret_ucb_threshold": thresholds[2],
                            "eligible": eligible,
                            "metrics": metrics,
                        }
                    )
    feasible = [row for row in frontier if row["eligible"]]
    diagnostic = max(
        (row for row in frontier if row["method"] != "mean"),
        key=lambda row: (
            row["metrics"]["deployment_precision"],
            -row["metrics"]["safety_violation_count"],
            row["metrics"]["intervention_count"],
        ),
    )
    if feasible:
        selected_entry = max(
            feasible,
            key=lambda row: (
                row["metrics"]["coverage"],
                row["metrics"]["precision_wilson_95_lower"],
                row["metrics"]["deployment_precision"],
            ),
        )
        safe_abstention = False
    else:
        selected_entry = {
            **diagnostic,
            "traffic_lcb_threshold": 1.000001,
            "metrics": deployment_metrics(roles["validation"], [False] * len(roles["validation"])),
            "eligible": False,
        }
        safe_abstention = True
    selected_alpha = (
        str(selected_entry["conformal_miscoverage"])
        if selected_entry["conformal_miscoverage"] is not None
        else str(config["conformal_miscoverage"][0])
    )
    selected_conformal = conformal_by_alpha[selected_alpha]
    policy = UpliftPolicy(
        selected_traffic,
        safety_model,
        regret_model,
        traffic_bootstrap,
        safety_bootstrap,
        regret_bootstrap,
        selected_conformal,
        selected_entry["method"],
        float(selected_entry["traffic_lcb_threshold"]),
        float(selected_entry["safety_ucb_threshold"]),
        float(selected_entry["regret_ucb_threshold"]),
    )

    selected_decisions = policy.decide(roles["validation"])
    selected_mask = np.asarray([row["intervene"] for row in selected_decisions])
    selected_prediction = selected_traffic.predict(values["validation"]["matrix"])
    comparison = {"V6_style": _v6_style_metrics(roles["validation"])}
    for method, alpha in methods:
        key = f"{method}|{alpha}"
        bounds = validation_bounds[key]
        mask = _mask(
            roles["validation"], bounds, (0.01, 0.25, 0.08)
        )
        label = (
            "V7_mean"
            if method == "mean"
            else "V7_quantile"
            if method == "bootstrap_quantile"
            else f"V7_conformal_{alpha}"
        )
        comparison[label] = deployment_metrics(roles["validation"], mask)
    direct_traffic_lower = direct_quantiles["traffic_q10"].predict(
        values["validation"]["matrix"]
    )
    direct_safety_upper = direct_quantiles["safety_q90"].predict(
        values["validation"]["matrix"]
    )
    direct_regret_upper = direct_quantiles["regret_q90"].predict(
        values["validation"]["matrix"]
    )
    direct_quantile_mask = np.asarray(
        [
            direct_traffic_lower[index] > 0.01
            and direct_safety_upper[index] <= 0.25
            and direct_regret_upper[index] <= 0.08
            and bool(row["outcomes"]["legal"])
            for index, row in enumerate(roles["validation"])
        ],
        dtype=bool,
    )
    comparison["V7_direct_pinball_quantile"] = deployment_metrics(
        roles["validation"], direct_quantile_mask
    )
    comparison["V7_frozen_candidate"] = deployment_metrics(
        roles["validation"], selected_mask
    )

    rng = np.random.default_rng(int(config["placebo_seed"]))
    placebo_target = rng.permutation(values["train"]["traffic"])
    placebo_values = {**values["train"], "traffic": placebo_target}
    placebo_model = _fit_traffic(
        selected_traffic.formulation,
        selected_traffic.kind,
        placebo_values,
        tuple(UPLIFT_V7_FEATURE_SCHEMA),
        seed + 5000,
    )
    placebo_prediction = placebo_model.predict(values["validation"]["matrix"])
    placebo = {
        "permutation_seed": int(config["placebo_seed"]),
        "real_signal": regression_metrics(
            values["validation"]["traffic"], selected_prediction
        ),
        "permuted_target": regression_metrics(
            values["validation"]["traffic"], placebo_prediction
        ),
        "expected_interpretation": "permuted treatment effects should destroy ranking signal",
    }

    ablations = []
    ablation_groups = {
        "traffic_temporal": UPLIFT_V7_FEATURE_GROUPS["traffic_temporal"],
        "analytical": UPLIFT_V7_FEATURE_GROUPS["analytical"],
        "v6_score": UPLIFT_V7_FEATURE_GROUPS["v6_score"],
        "topology": (
            *UPLIFT_V7_FEATURE_GROUPS["topology"],
            *UPLIFT_V7_FEATURE_GROUPS["topology_v7"],
        ),
        "preference": (
            *UPLIFT_V7_FEATURE_GROUPS["preference"],
            *UPLIFT_V7_FEATURE_GROUPS["preference_v7"],
        ),
        "penetration": ("navigation_penetration", "acceptance_multiplier"),
        "safety": UPLIFT_V7_FEATURE_GROUPS["safety"],
    }
    for index, name in enumerate(config["feature_ablations"]):
        removed = set(ablation_groups[name])
        names = tuple(item for item in UPLIFT_V7_FEATURE_SCHEMA if item not in removed)
        train_values = _arrays(roles["train"], names)
        validation_values = _arrays(roles["validation"], names)
        model = _fit_traffic(
            selected_traffic.formulation,
            selected_traffic.kind,
            train_values,
            names,
            seed + 6000 + index,
        )
        ablations.append(
            {
                "ablation": f"without_{name}",
                "removed_features": sorted(removed),
                "remaining_feature_count": len(names),
                "validation_metrics": regression_metrics(
                    validation_values["traffic"],
                    model.predict(validation_values["matrix"]),
                ),
            }
        )

    _write(MODEL_OUTPUT / "candidate_comparison.json", candidates)
    _write(
        MODEL_OUTPUT / "direct_quantile_models.json",
        {
            "models": {name: model.to_dict() for name, model in direct_quantiles.items()},
            "validation": {
                "traffic_q10_empirical_coverage": float(
                    np.mean(
                        values["validation"]["traffic"]
                        >= direct_quantiles["traffic_q10"].predict(
                            values["validation"]["matrix"]
                        )
                    )
                ),
                "safety_q90_empirical_coverage": float(
                    np.mean(
                        values["validation"]["safety"]
                        <= direct_quantiles["safety_q90"].predict(
                            values["validation"]["matrix"]
                        )
                    )
                ),
                "regret_q90_empirical_coverage": float(
                    np.mean(
                        values["validation"]["regret"]
                        <= direct_quantiles["regret_q90"].predict(
                            values["validation"]["matrix"]
                        )
                    )
                ),
                "deployment_metrics": comparison["V7_direct_pinball_quantile"],
            },
        },
    )
    _write(MODEL_OUTPUT / "auxiliary_comparison.json", auxiliary_comparison)
    _write(
        MODEL_OUTPUT / "effect_calibration.json",
        effect_calibration(values["validation"]["traffic"], selected_prediction),
    )
    _write(
        MODEL_OUTPUT / "cumulative_gain.json",
        cumulative_gain(values["validation"]["traffic"], selected_prediction),
    )
    _write(MODEL_OUTPUT / "placebo.json", placebo)
    _write(MODEL_OUTPUT / "ablation.json", ablations)
    _write(
        MODEL_OUTPUT / "training_manifest.json",
        {
            "selection_data": "v7_development_only",
            "pair_count": len(rows),
            "role_counts": {name: len(value) for name, value in roles.items()},
            "case_ids": {
                name: sorted(row["pair_id"] for row in value) for name, value in roles.items()
            },
            "selected_traffic": selected_traffic_entry,
            "selected_auxiliary": selected_auxiliary,
            "final_holdout_used": False,
            "rl_used": False,
        },
    )
    _write(
        MODEL_OUTPUT / "scenario_family_holdouts.json",
        _scenario_family_holdouts(
            rows, selected_traffic.formulation, selected_traffic.kind, seed
        ),
    )
    _write(POLICY_OUTPUT / "precision_coverage_frontier.json", frontier)
    _write(POLICY_OUTPUT / "policy_comparison.json", comparison)
    _write(
        POLICY_OUTPUT / "scenario_grouped_validation.json",
        _scenario_metrics(roles["validation"], selected_mask),
    )
    _write(
        package_path,
        {
            "version": "concordia-v7-development-selected-1",
            "policy": policy.to_dict(),
            "safe_abstention": safe_abstention,
            "selection": selected_entry,
            "best_diagnostic": diagnostic,
            "final_holdout_used": False,
        },
    )
    _write(
        summary_path,
        {
            "complete": True,
            "selection_objective": "maximize coverage subject to precision>=0.80, safety=0, support>=15",
            "selected_traffic": selected_traffic_entry,
            "selected_auxiliary": selected_auxiliary,
            "selected_policy": selected_entry,
            "best_diagnostic": diagnostic,
            "safe_abstention": safe_abstention,
            "validation_metrics": deployment_metrics(roles["validation"], selected_mask),
            "traffic_effect_metrics": regression_metrics(
                values["validation"]["traffic"], selected_prediction
            ),
            "effect_calibration": effect_calibration(
                values["validation"]["traffic"], selected_prediction
            ),
            "policy_comparison": comparison,
            "placebo": placebo,
            "wilson_lcb_secondary_tiebreak": True,
            "final_holdout_used": False,
            "rl_used": False,
        },
    )
    print(package_path)
    return package_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    run(force=arguments.force)


if __name__ == "__main__":
    main()
