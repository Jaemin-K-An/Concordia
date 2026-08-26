from __future__ import annotations

import unittest

from concordia.v9.action_features import ACTION_FEATURE_SCHEMA, build_action_features
from concordia.v9.action_space import (
    INTENSITIES,
    ROUTE_ALLOCATIONS,
    USER_STRATEGIES,
    generate_action_library,
)
from concordia.v9.optimizer import RobustActionOptimizer
from concordia.v9.oracle import oracle_actionability
from concordia.v9.rollout import assert_seed_disjoint, rollout_seeds


class V9MultiActionTests(unittest.TestCase):
    def test_balanced_library_contains_null_and_all_registered_factors(self):
        actions = generate_action_library()
        self.assertEqual(len(actions), 25)
        self.assertTrue(actions[0].is_null)
        self.assertEqual({action.reroute_fraction for action in actions[1:]}, set(INTENSITIES))
        self.assertEqual({action.user_strategy for action in actions[1:]}, set(USER_STRATEGIES))
        self.assertEqual({action.route_allocation for action in actions[1:]}, set(ROUTE_ALLOCATIONS))

    def test_action_features_are_exact_and_predecision_only(self):
        action = generate_action_library()[3].to_dict()
        features = build_action_features(action, {
            "proposed_rerouted_user_count": 4,
            "expected_accepted_user_count": 2.5,
            "expected_rerouted_flow": 75,
            "destination_capacity_slack": 0.8,
        })
        self.assertEqual(tuple(features), ACTION_FEATURE_SCHEMA)
        self.assertEqual(features["reroute_fraction"], action["reroute_fraction"])
        self.assertFalse(any("realized" in name or "tau_" in name for name in features))

    def test_placebo_fixture_abstains(self):
        selected = RobustActionOptimizer().select([
            {"action_id": "A00_NULL_B1", "is_null": True, "robust_benefit": 0.0},
            {
                "action_id": "A01", "is_null": False, "robust_benefit": 0.0,
                "ml_unsafe_probability": 0.0, "rollout_unsafe_probability": 0.0,
                "predicted_max_regret": 0.0, "legal": True,
            },
        ])
        self.assertFalse(selected["intervene"])
        self.assertEqual(selected["action_id"], "A00_NULL_B1")

    def test_strong_safe_fixture_selects_highest_robust_value(self):
        selected = RobustActionOptimizer().select([
            {"action_id": "A00_NULL_B1", "is_null": True, "robust_benefit": 0.0},
            {
                "action_id": "A01", "is_null": False, "robust_benefit": 0.02,
                "ml_unsafe_probability": 0.01, "rollout_unsafe_probability": 0.02,
                "predicted_max_regret": 0.02, "legal": True,
            },
            {
                "action_id": "A02", "is_null": False, "robust_benefit": 0.08,
                "ml_unsafe_probability": 0.01, "rollout_unsafe_probability": 0.20,
                "predicted_max_regret": 0.02, "legal": True,
            },
        ])
        self.assertTrue(selected["intervene"])
        self.assertEqual(selected["action_id"], "A01")

    def test_oracle_and_rollout_seed_contracts(self):
        rows = [
            {"state_id": "s1", "action_id": "A00_NULL_B1", "outcomes": {
                "tau_t_relative": 0.0, "tau_s": 0.0, "max_regret": 0.0, "legal": True,
            }},
            {"state_id": "s1", "action_id": "A01", "outcomes": {
                "tau_t_relative": 0.02, "tau_s": 0.0, "max_regret": 0.01, "legal": True,
            }},
        ]
        report = oracle_actionability(rows)
        self.assertEqual(report["oracle_actionability_rate"], 1.0)
        seeds = rollout_seeds("s1", "A01", 5, realized_seed=1409)
        self.assertEqual(len(set(seeds)), 5)
        assert_seed_disjoint([1409], seeds)


if __name__ == "__main__":
    unittest.main()
