from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import numpy as np

from concordia.evaluation import adaptive_success_claim_allowed
from concordia.feasibility import (
    RegimeDefinition,
    RobustShiftDetector,
    unified_calibration_metrics,
)
from concordia.selective import RegimeConditionedPolicy, V5DecisionInputs


ROOT = Path(__file__).resolve().parents[1]


class V5RegimeShiftTests(unittest.TestCase):
    def test_regime_boundary_and_routing_are_deterministic(self):
        definition = RegimeDefinition(0.4, 0.8, 0.75, 0.8, 1.0)
        features = {
            "navigation_penetration": 1.0,
            "route_overlap": 0.2,
            "alternative_capacity_ratio": 1.2,
        }
        self.assertEqual(definition.route(features), "HIGH_CONTROL")
        self.assertEqual(definition.route(features), "HIGH_CONTROL")
        features["navigation_penetration"] = 0.5
        self.assertEqual(definition.route(features), "PARTIAL_CONTROL")
        features.update({"route_overlap": 0.9, "alternative_capacity_ratio": 0.5})
        self.assertEqual(definition.route(features), "STRUCTURALLY_CONSTRAINED")

    def test_known_ood_point_has_higher_dss(self):
        rng = np.random.default_rng(4)
        training = rng.normal(0.0, 0.2, (200, 5))
        detector = RobustShiftDetector.fit(training)
        score = detector.score(np.vstack([np.zeros(5), np.full(5, 8.0)]))
        self.assertGreater(score[1], score[0])
        self.assertEqual(detector.classify(np.full((1, 5), 8.0))[0], "STRONG_SHIFT")

    def test_calibration_protocol_is_single_and_bounded(self):
        metrics = unified_calibration_metrics(
            [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]
        )
        self.assertEqual(metrics["bin_count"], 10)
        self.assertIn("equal-width", metrics["protocol"])
        self.assertGreaterEqual(metrics["ece"], 0.0)

    def test_micro_safety_veto_never_intervenes_when_unsafe(self):
        policy = RegimeConditionedPolicy(
            {"HIGH_CONTROL": {"IN_DISTRIBUTION": 0.7}},
            shift_probability_penalty=0.08,
            micro_success_threshold=0.5,
            micro_safety_threshold=0.1,
        )
        decision = policy.decide(
            V5DecisionInputs(
                "unsafe",
                "HIGH_CONTROL",
                "IN_DISTRIBUTION",
                0.1,
                0.95,
                0.05,
                0.04,
                0.9,
                0.8,
                True,
            )
        )
        self.assertFalse(decision.intervene)
        self.assertIn("microscopic safety veto", decision.reasons)

    def test_strong_shift_always_abstains(self):
        policy = RegimeConditionedPolicy(
            {"HIGH_CONTROL": {"STRONG_SHIFT": 0.5}},
            0.08,
            0.5,
            0.2,
        )
        inputs = V5DecisionInputs(
            "ood", "HIGH_CONTROL", "STRONG_SHIFT", 1.4, 0.99, 0.1, 0.1, 0.99, 0.0, True
        )
        self.assertFalse(policy.decide(inputs).intervene)

    def test_zero_success_forbids_adaptive_claim(self):
        metrics = {
            "intervention_count": 20,
            "successful_intervention_count": 0,
            "safety_violation_count": 0,
        }
        self.assertFalse(adaptive_success_claim_allowed(metrics, 10))
        metrics["successful_intervention_count"] = 1
        metrics["safety_violation_count"] = 1
        self.assertFalse(adaptive_success_claim_allowed(metrics, 10))

    def test_v4_failed_micro_fixture_is_vetoed(self):
        summary = json.loads(
            (ROOT / "artifacts/studies/v4_microscopic/summary.json").read_text()
        )
        self.assertEqual(summary["policy_metrics"]["V4-F"]["successful_intervention_count"], 0)
        self.assertGreater(summary["policy_metrics"]["V4-F"]["safety_violation_count"], 0)
        policy = RegimeConditionedPolicy(
            {"PARTIAL_CONTROL": {"IN_DISTRIBUTION": 0.7}}, 0.08, 0.5, 0.1
        )
        decision = policy.decide(
            V5DecisionInputs(
                "v4-micro-failure",
                "PARTIAL_CONTROL",
                "IN_DISTRIBUTION",
                0.2,
                0.95,
                0.04,
                0.03,
                0.8,
                0.9,
                True,
            )
        )
        self.assertFalse(decision.intervene)

    def test_v5_split_config_checksum_read_is_immutable(self):
        path = ROOT / "configs/v5/splits.yaml"
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        _ = path.read_text()
        after = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
