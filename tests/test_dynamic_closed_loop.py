import unittest

from concordia.adaptive import DynamicRoutePredictor, NetworkStateEstimator, dynamic_preference_slack
from concordia.adaptive.controller import ClosedLoopController
from concordia.behavior import AcceptanceCoefficients, AcceptanceModel
from concordia.models import PreferenceVector, User
from concordia.optimization import MIPAssignmentSolver, RecedingHorizonOptimizer
from concordia.preferences import UtilityModel
from concordia.scenarios import two_route
from concordia.simulation import AnalyticalSimulationAdapter, analytical_edge_id


class DynamicClosedLoopTests(unittest.TestCase):
    def setUp(self):
        self.network, self.od, self.demand = two_route()
        route_list = self.network.multiobjective_candidate_routes(
            *self.od,
            k_per_objective=2,
            max_overlap=1.0,
            pareto_filter=False,
        )
        self.routes = {route.route_id: route for route in route_list}
        self.route_ids = tuple(self.routes)
        self.users = [
            User(
                f"u{index}",
                *self.od,
                PreferenceVector(0.5, 0.2, 0.0, 0.1, 0.1, 0.1),
                epsilon=1.0,
            )
            for index in range(4)
        ]
        self.candidates = {user.user_id: self.route_ids for user in self.users}
        self.initial = {user.user_id: self.route_ids[0] for user in self.users}
        self.acceptance = AcceptanceModel(
            AcceptanceCoefficients(
                intercept=100.0,
                preference_slack=0.0,
                utility_gain=0.0,
                eta_gain_minutes=0.0,
                reliability_gain_minutes2=0.0,
                network_benefit=0.0,
                source="synthetic_test",
            )
        )

    def test_multiobjective_generation_covers_distinct_tradeoffs(self):
        paths = {route.nodes for route in self.routes.values()}
        self.assertEqual(paths, {("O", "A", "D"), ("O", "B", "D")})

    def test_dynamic_slack_uses_projected_congestion(self):
        congested_route = self.routes[self.route_ids[0]]
        target_flows = {edge: 0.0 for edge in self.network.edges}
        for edge in congested_route.edges:
            target_flows[edge] = self.demand
        state = NetworkStateEstimator(self.network).from_flows(
            {edge: 0.0 for edge in self.network.edges}, 0.0
        )
        predictor = DynamicRoutePredictor(self.network, horizon_steps=3, flow_relaxation=1.0)
        predictions = [
            predictor.predict(route, state, target_flows) for route in self.routes.values()
        ]
        slack = dynamic_preference_slack(self.users[0], predictions, UtilityModel())
        self.assertEqual(min(slack.values()), 0.0)
        self.assertGreater(
            predictions[0].expected_features.time,
            self.routes[self.route_ids[0]].features.time,
        )

    def test_receding_horizon_executes_first_action_then_replans(self):
        simulator = AnalyticalSimulationAdapter(
            self.network,
            self.routes,
            self.initial,
            self.demand / len(self.users),
        )
        optimizer = RecedingHorizonOptimizer(
            self.network,
            self.routes,
            self.demand / len(self.users),
            horizon_steps=3,
            acceptance_model=self.acceptance,
        )
        route_edge_ids = {
            route_id: [analytical_edge_id(edge) for edge in route.edges]
            for route_id, route in self.routes.items()
        }
        controller = ClosedLoopController(
            self.network,
            self.routes,
            self.users,
            self.candidates,
            self.initial,
            simulator,
            simulator.edge_id_map,
            optimizer,
            self.acceptance,
            seed=7,
            route_edge_ids=route_edge_ids,
        )
        result = controller.run(steps=2)
        self.assertEqual(len(result.steps), 2)
        self.assertEqual(result.acceptance_rate, 1.0)
        self.assertNotEqual(set(result.final_assignments.values()), {self.route_ids[0]})
        self.assertNotEqual(
            result.steps[0].plan.first_assignments,
            result.steps[1].plan.first_assignments,
        )
        self.assertTrue(
            all(
                step.plan.expected_safety_cvar
                <= step.plan.baseline_safety_cvar + optimizer.safety_delta + 1e-10
                for step in result.steps
            )
        )

    def test_mip_enforces_assignment_regret_acceptance_and_safety(self):
        utilities = {
            user.user_id: {route_id: 0.0 for route_id in self.route_ids}
            for user in self.users
        }
        costs = {
            (user.user_id, route_id): float(index + 1)
            for user in self.users
            for index, route_id in enumerate(self.route_ids)
        }
        probabilities = {key: 1.0 for key in costs}
        solver = MIPAssignmentSolver(self.network, self.routes, minimum_acceptance_probability=0.5)
        result = solver.solve(
            self.users,
            self.candidates,
            utilities,
            costs,
            probabilities,
            baseline_assignments=self.initial,
            safety_delta=0.0,
        )
        self.assertTrue(result.optimal)
        self.assertEqual(set(result.assignments.values()), {self.route_ids[0]})
        self.assertIn("HiGHS", result.solver)


if __name__ == "__main__":
    unittest.main()
