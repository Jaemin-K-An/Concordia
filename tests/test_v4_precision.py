from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from concordia.feasibility import (
    ConformalRiskController,
    FEATURE_SCHEMA,
    ProbabilityCalibrator,
    V4_FEATURE_SCHEMA,
    calibration_error,
    precision_constrained_threshold,
)
from concordia.selective import (
    PrecisionConstrainedPolicy,
    V4DecisionInputs,
    expected_safe_intervention_value,
    risk_adjusted_esiv,
)


ROOT = Path(__file__).resolve().parents[1]


class V4PrecisionTests(unittest.TestCase):
    def test_v4_schema_preserves_frozen_v3_prefix(self):
        self.assertEqual(V4_FEATURE_SCHEMA[: len(FEATURE_SCHEMA)], FEATURE_SCHEMA)
        self.assertIn("aps_alternative_capacity_interaction", V4_FEATURE_SCHEMA)

    def test_calibration_probabilities_are_bounded_monotone_and_improve_shift_fixture(self):
        source = np.linspace(0.03, 0.97, 200)
        labels = (source >= 0.62).astype(int)
        raw = np.sqrt(source)
        calibrator = ProbabilityCalibrator("isotonic").fit(raw[::2], labels[::2])
        calibrated = calibrator.predict(raw[1::2])
        self.assertTrue(np.all((calibrated >= 0.0) & (calibrated <= 1.0)))
        self.assertTrue(np.all(np.diff(calibrated) >= -1e-12))
        self.assertLessEqual(
            calibration_error(labels[1::2], calibrated)["ece"],
            calibration_error(labels[1::2], raw[1::2])["ece"],
        )

    def test_precision_constrained_optimizer_is_deterministic_and_guards_coverage(self):
        labels = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
        scores = [0.99, 0.95, 0.91, 0.82, 0.72, 0.61, 0.4, 0.3, 0.2, 0.1]
        kwargs = {
            "precision_target": 0.80,
            "coverage_guard": 0.20,
            "thresholds": [0.5, 0.8, 0.9],
        }
        first = precision_constrained_threshold(labels, scores, **kwargs)
        second = precision_constrained_threshold(labels, scores, **kwargs)
        self.assertEqual(first, second)
        self.assertGreaterEqual(first["selected"]["precision"], 0.80)
        self.assertGreaterEqual(first["selected"]["coverage"], 0.20)

    def test_esiv_is_nonnegative_and_decreases_with_safety_risk(self):
        low_risk = expected_safe_intervention_value(0.9, 0.04, 0.05)
        high_risk = expected_safe_intervention_value(0.9, 0.04, 0.40)
        self.assertGreater(low_risk, high_risk)
        self.assertGreaterEqual(risk_adjusted_esiv(0.7, -0.1, 0.2), 0.0)

    def test_predicted_unsafe_never_intervenes(self):
        policy = PrecisionConstrainedPolicy(
            "V4-P",
            probability_threshold=0.8,
            benefit_threshold=0.01,
            safety_delta=0.25,
            safety_probability_threshold=0.2,
        )
        decision = policy.decide(
            V4DecisionInputs(
                case_id="unsafe",
                success_probability=0.99,
                success_probability_lower=0.95,
                expected_benefit=0.1,
                benefit_lower=0.08,
                safety_difference_upper=0.30,
                safety_failure_probability=0.01,
                safety_failure_probability_upper=0.02,
                esiv=0.09,
                esiv_lower=0.07,
                legal=True,
            )
        )
        self.assertFalse(decision.intervene)
        self.assertEqual(decision.selected_policy, "B1_ETA_BASELINE")

    def test_conformal_risk_controller_is_deterministic(self):
        probabilities = np.linspace(0.01, 0.99, 100)
        labels = (probabilities >= 0.55).astype(int)
        controller = ConformalRiskController.fit(
            probabilities,
            labels,
            target_error_rate=0.20,
            thresholds=np.linspace(0.5, 0.95, 10),
        )
        restored = ConformalRiskController.from_dict(controller.to_dict())
        np.testing.assert_array_equal(
            controller.intervene(probabilities), restored.intervene(probabilities)
        )

    def test_final_holdout_seeds_are_absent_from_fitting_roles(self):
        split = yaml.safe_load((ROOT / "configs/v4/splits.yaml").read_text(encoding="utf-8"))
        fitting = set(split["roles"]["training_seeds"])
        fitting |= set(split["roles"]["calibration_fit_seeds"])
        fitting |= set(split["roles"]["calibration_evaluation_seeds"])
        fitting |= set(split["roles"]["validation_seeds"])
        self.assertFalse(fitting & set(split["final_holdout"]["seeds"]))
        expected = (
            len(split["final_holdout"]["scenarios"])
            * len(split["final_holdout"]["seeds"])
            * len(split["final_holdout"]["demand_scale"])
            * len(split["final_holdout"]["heterogeneity"])
            * len(split["final_holdout"]["navigation_penetration"])
        )
        self.assertEqual(expected, split["final_holdout"]["expected_case_count"])

    def test_freeze_checksum_read_is_immutable(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "frozen.json"
            path.write_text(json.dumps({"precision": 0.8}) + "\n", encoding="utf-8")
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            _ = json.loads(path.read_text(encoding="utf-8"))
            after = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
