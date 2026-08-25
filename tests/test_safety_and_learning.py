import unittest

import numpy as np

from concordia.learning import LinUCBPreferenceLearner
from concordia.safety import TrajectoryFrame, summarize_safety


class SafetyTests(unittest.TestCase):
    def test_ttc_drac_hard_braking_and_tail(self):
        frames = [
            TrajectoryFrame(0, "f1", "l1", 10, 20, 10, -5),
            TrajectoryFrame(1, "f2", "l2", 20, 20, 15, -1),
            TrajectoryFrame(2, "f3", None, None, 5, None, 0),
        ]
        summary = summarize_safety(frames, pet_values=[0.8, 2.0], ttc_threshold=1.5)
        self.assertEqual(summary.ttc_values, (1.0, 4.0))
        self.assertEqual(summary.drac_values, (5.0, 0.625))
        self.assertEqual(summary.ttc_conflicts, 1)
        self.assertEqual(summary.hard_braking_events, 1)
        self.assertGreaterEqual(summary.cvar_drac_95, max(summary.drac_values))


class BanditTests(unittest.TestCase):
    def test_learning_moves_estimate_toward_rewarded_feature(self):
        learner = LinUCBPreferenceLearner(2, alpha=0)
        for _ in range(20):
            learner.update([1, 0], 1)
            learner.update([0, 1], 0)
        self.assertGreater(learner.estimate[0], learner.estimate[1])
        self.assertEqual(learner.choose({"good": [1, 0], "bad": [0, 1]}), "good")
        self.assertTrue(np.all(np.isfinite(learner.estimate)))


if __name__ == "__main__":
    unittest.main()
