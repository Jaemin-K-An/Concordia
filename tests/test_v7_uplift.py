from __future__ import annotations

import unittest

import numpy as np

from concordia.errors import ValidationError
from concordia.feasibility import build_alignment_case
from concordia.micro_v6.features import MICRO_V6_FEATURE_SCHEMA
from concordia.uplift_v7.conformal import ConformalAdjustments
from concordia.uplift_v7.evaluation import deployment_metrics
from concordia.uplift_v7.learners import RegressionModel
from concordia.uplift_v7.outcomes import paired_treatment_outcomes
from concordia.uplift_v7.paired_dataset import (
    UPLIFT_V7_FEATURE_SCHEMA,
    paired_row_from_v6,
    validate_predecision_features,
)
from concordia.uplift_v7.policy import UpliftPolicy
from concordia.uplift_v7.quantiles import PinballQuantileRegressor


class _ConstantModel:
    def __init__(self, value: float):
        self.value = value
        self.feature_names = ("x",)

    def predict(self, matrix):
        return np.full(len(matrix), self.value, dtype=float)


class _ConstantEnsemble:
    def __init__(self, lower: float, upper: float):
        self.lower = lower
        self.upper = upper

    def interval(self, matrix, _lower_quantile, _upper_quantile):
        mean = np.full(len(matrix), (self.lower + self.upper) / 2.0)
        return mean, np.full(len(matrix), self.lower), np.full(len(matrix), self.upper)


class V7UpliftTest(unittest.TestCase):
    @staticmethod
    def _run(ttt: float, risk: float, regret: float = 0.02) -> dict:
        return {
            "total_travel_time_seconds": ttt,
            "maximum_affected_regret": regret,
            "all_executed_routes_legal": True,
            "safety": {"cvar_drac_95": risk},
        }

    def test_paired_effect_signs_follow_preregistration(self):
        outcome = paired_treatment_outcomes(
            self._run(1000.0, 1.0), self._run(900.0, 1.1)
        )
        self.assertEqual(outcome.tau_t_seconds, 100.0)
        self.assertAlmostEqual(outcome.tau_t_relative, 0.10)
        self.assertAlmostEqual(outcome.tau_s, 0.10)
        self.assertTrue(outcome.safe_micro_success)

    def test_pair_metadata_is_identical_except_treatment(self):
        baseline = {
            **self._run(1000.0, 1.0),
            "generated_vehicle_count": 10,
        }
        adaptive = {
            **self._run(900.0, 1.1),
            "generated_vehicle_count": 10,
        }
        row = {
            "case_id": "fixture",
            "seed": 7,
            "condition": {"topology": "two_route"},
            "decision_time": 30.0,
            "feature_observation_end_time": 30.0,
            "features_pre_decision": {
                name: 0.0 for name in MICRO_V6_FEATURE_SCHEMA
            },
            "counterfactual_B1": baseline,
            "counterfactual_adaptive": adaptive,
            "pairing": {
                "same_seed": True,
                "same_network_hash": True,
                "same_route_file_hash": True,
                "common_random_numbers": True,
            },
        }
        paired = paired_row_from_v6(
            row, v6_micro_success_score=0.2, source="fixture"
        )
        self.assertTrue(paired["pairing"]["metadata_identical_except_treatment"])
        self.assertTrue(paired["pairing"]["treatment_only_difference"])

    def test_zero_effect_is_not_a_success(self):
        outcome = paired_treatment_outcomes(
            self._run(1000.0, 1.0), self._run(1000.0, 1.0)
        )
        self.assertEqual(outcome.tau_t_relative, 0.0)
        self.assertFalse(outcome.safe_micro_success)

    def test_strong_effect_fixture_is_a_success(self):
        outcome = paired_treatment_outcomes(
            self._run(1000.0, 1.0), self._run(800.0, 1.2, 0.08)
        )
        self.assertTrue(outcome.safe_micro_success)
        self.assertEqual(outcome.benefit_magnitude_bin, "strong_above_5_percent")

    def test_handcrafted_high_capacity_analytical_fixture_has_positive_effect(self):
        case = build_alignment_case(
            scenario="two_route",
            seed=7,
            demand_scale=0.7,
            heterogeneity="low",
            navigation_penetration=1.0,
            user_count=6,
            regret_limit=0.08,
            epsilon_grid=[0.0, 0.02, 0.04, 0.08, 0.12, 0.16],
            minimum_relative_ttt_gain=0.01,
            safety_delta=0.25,
            source_split="v7_strong_effect_fixture_only",
        )
        baseline = case["baseline_metrics"]["eta_only_ttt"]
        adaptive = case["adaptive_counterfactual"]["ttt"]
        self.assertGreater((baseline - adaptive) / baseline, 0.05)

    def test_feature_schema_rejects_post_treatment_name(self):
        row = {name: 0.0 for name in UPLIFT_V7_FEATURE_SCHEMA}
        row["realized_tau_t"] = 1.0
        with self.assertRaises(ValidationError):
            validate_predecision_features(row)

    def test_policy_requires_all_three_uncertainty_gates(self):
        policy = UpliftPolicy(
            _ConstantModel(0.03),
            _ConstantModel(0.05),
            _ConstantModel(0.03),
            _ConstantEnsemble(0.02, 0.04),
            _ConstantEnsemble(0.00, 0.10),
            _ConstantEnsemble(0.01, 0.05),
            ConformalAdjustments(0.1, 0.01, 0.05, 0.02),
            "bootstrap_quantile",
            0.01,
            0.25,
            0.08,
        )
        row = {"pair_id": "p", "predecision_features": {"x": 0.0}}
        self.assertTrue(policy.decide([row])[0]["intervene"])
        policy.safety_bootstrap = _ConstantEnsemble(0.20, 0.30)
        decision = policy.decide([row])[0]
        self.assertFalse(decision["intervene"])
        self.assertEqual(decision["reason"], "safety_effect_veto")

    def test_traffic_lcb_below_minimum_forces_b1(self):
        policy = UpliftPolicy(
            _ConstantModel(0.01),
            _ConstantModel(0.0),
            _ConstantModel(0.0),
            _ConstantEnsemble(0.005, 0.015),
            _ConstantEnsemble(0.0, 0.0),
            _ConstantEnsemble(0.0, 0.0),
            ConformalAdjustments(0.1, 0.01, 0.0, 0.0),
            "bootstrap_quantile",
            0.01,
            0.25,
            0.08,
        )
        row = {"pair_id": "p", "predecision_features": {"x": 0.0}}
        decision = policy.decide([row])[0]
        self.assertFalse(decision["intervene"])
        self.assertEqual(decision["executed_policy"], "B1")

    def test_regression_model_round_trip(self):
        matrix = np.arange(40, dtype=float).reshape(20, 2)
        target = matrix[:, 0] * 0.2 - matrix[:, 1] * 0.1
        fitted = RegressionModel("fixture", "ridge", ("a", "b")).fit(matrix, target)
        restored = RegressionModel.from_dict(fitted.to_dict())
        np.testing.assert_allclose(fitted.predict(matrix), restored.predict(matrix))

    def test_direct_quantile_model_round_trip(self):
        matrix = np.arange(60, dtype=float).reshape(30, 2)
        target = matrix[:, 0] * 0.01
        fitted = PinballQuantileRegressor(0.10, ("a", "b"), iterations=200).fit(
            matrix, target
        )
        restored = PinballQuantileRegressor.from_dict(fitted.to_dict())
        np.testing.assert_allclose(fitted.predict(matrix), restored.predict(matrix))

    def test_deployment_metrics_count_only_selected_failures(self):
        rows = [
            {
                "outcomes": {
                    "safe_micro_success": True,
                    "tau_t_relative": 0.04,
                    "tau_s": 0.0,
                    "max_regret": 0.01,
                }
            },
            {
                "outcomes": {
                    "safe_micro_success": False,
                    "tau_t_relative": -0.02,
                    "tau_s": 0.5,
                    "max_regret": 0.10,
                }
            },
        ]
        metrics = deployment_metrics(rows, [True, False])
        self.assertEqual(metrics["intervention_count"], 1)
        self.assertEqual(metrics["deployment_precision"], 1.0)
        self.assertEqual(metrics["safety_violation_count"], 0)


if __name__ == "__main__":
    unittest.main()
