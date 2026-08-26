from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from concordia.feasibility.calibration_v4 import ProbabilityCalibrator
from concordia.safety_v8.classifier import SafetyClassifier
from concordia.safety_v8.features import (
    ACTION_AWARE_FEATURE_SCHEMA,
    FORBIDDEN_INPUT_TOKENS,
    action_aware_features,
)
from concordia.safety_v8.labels import unsafe_intervention


ROOT = Path(__file__).resolve().parents[1]


class V8SafetyTests(unittest.TestCase):
    def test_registered_unsafe_label_is_strict(self):
        self.assertFalse(unsafe_intervention({"risk_b1": 1.0, "risk_adaptive": 1.25}))
        self.assertTrue(unsafe_intervention({"risk_b1": 1.0, "risk_adaptive": 1.250001}))

    def test_action_schema_has_no_forbidden_postdecision_names(self):
        self.assertFalse(
            [
                name for name in ACTION_AWARE_FEATURE_SCHEMA
                if any(token in name.lower() for token in FORBIDDEN_INPUT_TOKENS)
            ]
        )

    def test_action_features_are_finite_and_complete(self):
        path = ROOT / "artifacts/studies/v7_paired_dataset/raw_metrics.json"
        row = json.loads(path.read_text())[0]
        features = action_aware_features(
            row, traffic_uplift_score=0.02, traffic_rank_percentile=0.80
        )
        self.assertEqual(set(features), set(ACTION_AWARE_FEATURE_SCHEMA))
        self.assertTrue(np.isfinite(list(features.values())).all())

    def test_classifier_and_calibrator_round_trip(self):
        rng = np.random.default_rng(8)
        matrix = rng.normal(size=(80, 5))
        labels = (matrix[:, 0] + matrix[:, 1] > 0.5).astype(int)
        model = SafetyClassifier("test", "logistic", tuple(f"x{i}" for i in range(5)), 8, 2.0)
        model.fit(matrix, labels)
        restored = SafetyClassifier.from_dict(model.to_dict())
        np.testing.assert_allclose(model.predict_proba(matrix), restored.predict_proba(matrix))
        calibrator = ProbabilityCalibrator("platt").fit(model.predict_proba(matrix), labels)
        calibrated = calibrator.predict(model.predict_proba(matrix))
        self.assertTrue(((calibrated >= 0.0) & (calibrated <= 1.0)).all())


if __name__ == "__main__":
    unittest.main()
