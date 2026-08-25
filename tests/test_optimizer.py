import unittest

from concordia.errors import ValidationError
from concordia.models import EdgeData, PreferenceVector, User
from concordia.network import RoadNetwork
from concordia.optimization import AdaptiveOptimizer
from concordia.scenarios import two_route


def users(count, od, epsilon):
    return [
        User(
            f"u{index}",
            od[0],
            od[1],
            PreferenceVector(0.5, 0.2, 0.0, 0.1, 0.1, 0.1),
            epsilon=epsilon,
        )
        for index in range(count)
    ]


class OptimizerTests(unittest.TestCase):
    def setUp(self):
        self.network, self.od, demand = two_route()
        route_list = self.network.candidate_routes(*self.od, k=2, max_overlap=1.0)
        self.routes = {route.route_id: route for route in route_list}
        self.candidates = {f"u{index}": tuple(self.routes) for index in range(6)}
        self.optimizer = AdaptiveOptimizer(self.network, self.routes, vehicle_flow=demand / 6)

    def test_exact_oracle_is_no_worse_than_greedy(self):
        population = users(6, self.od, epsilon=1.0)
        exact = self.optimizer.exact(population, self.candidates, safety_delta=0)
        greedy = self.optimizer.greedy_vde(population, self.candidates, safety_delta=0)
        self.assertLessEqual(exact.objective, greedy.objective + 1e-8)
        self.assertTrue(all(exact.regrets[user.user_id] <= user.epsilon for user in population))

    def test_no_sacrifice_consumes_no_slack(self):
        population = users(6, self.od, epsilon=0.0)
        exact = self.optimizer.exact(population, self.candidates)
        self.assertLessEqual(max(exact.regrets.values()), 1e-10)

    def test_exact_guard_rejects_unbounded_enumeration(self):
        population = users(6, self.od, epsilon=1.0)
        with self.assertRaises(ValidationError):
            self.optimizer.exact(population, self.candidates, max_combinations=10)

    def test_hard_safety_constraint_rejects_riskier_improvement(self):
        network = RoadNetwork("safety")
        network.add_edge("O", "A", EdgeData(1, 100, risk=0.0))
        network.add_edge("A", "D", EdgeData(10, 100, risk=0.0))
        network.add_edge("O", "B", EdgeData(1, 100, risk=1.0))
        network.add_edge("B", "D", EdgeData(1, 100, risk=1.0))
        route_list = network.candidate_routes("O", "D", k=2, max_overlap=1.0)
        route_map = {route.route_id: route for route in route_list}
        user = User("u", "O", "D", PreferenceVector(0, 0, 0, 1, 0, 0), epsilon=10)
        optimizer = AdaptiveOptimizer(network, route_map, vehicle_flow=1)
        result = optimizer.exact([user], {"u": tuple(route_map)}, safety_delta=0)
        self.assertEqual(result.total_safety_risk, 0.0)


if __name__ == "__main__":
    unittest.main()
