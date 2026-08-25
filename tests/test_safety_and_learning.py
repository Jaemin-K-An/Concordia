import unittest

import numpy as np

from concordia.behavior import DuelingPreferenceLearner, PopulationPrior, UserPreferencePosterior
from concordia.learning import LinUCBPreferenceLearner
from concordia.safety import TrajectoryFrame, safety_non_degradation, summarize_safety


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
        self.assertEqual(summary.high_closing_speed_conflicts, 1)
        self.assertEqual(summary.observation_count, 3)
        self.assertGreaterEqual(summary.p99_drac, summary.p95_drac)

    def test_tail_non_degradation_detects_worse_proposed_risk(self):
        baseline = summarize_safety([TrajectoryFrame(0, "f", "l", 20, 15, 10)])
        proposed = summarize_safety([TrajectoryFrame(0, "f", "l", 5, 25, 5)])
        comparison = safety_non_degradation(baseline, proposed, delta=0.0)
        self.assertFalse(comparison.passed)
        self.assertIn("cvar_drac_95", comparison.reasons)


class BanditTests(unittest.TestCase):
    def test_learning_moves_estimate_toward_rewarded_feature(self):
        learner = LinUCBPreferenceLearner(2, alpha=0)
        for _ in range(20):
            learner.update([1, 0], 1)
            learner.update([0, 1], 0)
        self.assertGreater(learner.estimate[0], learner.estimate[1])
        self.assertEqual(learner.choose({"good": [1, 0], "bad": [0, 1]}), "good")
        self.assertTrue(np.all(np.isfinite(learner.estimate)))

    def test_population_prior_posterior_dueling_and_drift(self):
        posterior = UserPreferencePosterior(PopulationPrior.synthetic_default(), forgetting_factor=0.95)
        learner = DuelingPreferenceLearner(posterior)
        candidates = [
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ]
        first, second = learner.choose_pair(candidates)
        learner.update(candidates[first], candidates[second], chose_first=True)
        self.assertEqual(posterior.observations, 1)
        before_risk = posterior.mean[3]
        posterior.apply_drift([1, 1, 1, 2, 1, 1])
        self.assertGreater(posterior.mean[3], before_risk)
        self.assertAlmostEqual(sum(posterior.preference_vector().as_dict().values()), 1.0)


if __name__ == "__main__":
    unittest.main()
