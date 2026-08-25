import math
import random
import unittest

from concordia.errors import ValidationError
from concordia.models import FeatureScales, PreferenceVector, Route, RouteFeatures
from concordia.preferences import UtilityModel, preference_slack, route_choice_probabilities


class UtilityTests(unittest.TestCase):
    def setUp(self):
        self.preference = PreferenceVector(4, 2, 1, 1, 1, 1)
        self.model = UtilityModel(FeatureScales(10, 10, 10, 1, 1, 1))

    def route(self, route_id, time, risk=0.1):
        return Route(route_id, ("O", route_id, "D"), RouteFeatures(time=time, risk=risk))

    def test_time_monotonicity(self):
        fast = self.route("fast", 8)
        slow = self.route("slow", 12)
        self.assertGreater(self.model.utility(self.preference, fast), self.model.utility(self.preference, slow))

    def test_preference_slack_nonnegative_and_best_zero(self):
        values = {"a": -1.0, "b": -2.25, "c": -1.0}
        slack = preference_slack(values)
        self.assertEqual(slack["a"], 0.0)
        self.assertEqual(slack["c"], 0.0)
        self.assertGreaterEqual(min(slack.values()), 0.0)

    def test_softmax_normalizes_for_random_utilities(self):
        rng = random.Random(42)
        for _ in range(100):
            utilities = {str(index): rng.uniform(-100, 100) for index in range(8)}
            probabilities = route_choice_probabilities(utilities, rationality=rng.uniform(0, 50))
            self.assertTrue(math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-12))
            self.assertTrue(all(0 <= value <= 1 for value in probabilities.values()))

    def test_zero_rationality_is_uniform(self):
        probabilities = route_choice_probabilities({"a": -99, "b": 4}, rationality=0)
        self.assertEqual(probabilities, {"a": 0.5, "b": 0.5})

    def test_invalid_route_features_fail_fast(self):
        with self.assertRaises(ValidationError):
            RouteFeatures(time=-1)


if __name__ == "__main__":
    unittest.main()
