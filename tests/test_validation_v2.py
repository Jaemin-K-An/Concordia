import json
import unittest
from pathlib import Path

import numpy as np

from concordia.alignment import compute_alignment_frontier
from concordia.optimization import PPOEligibilityPolicy, solve_fixed_point
from concordia.populations import generate_population
from concordia.scenarios import two_route
from concordia.traffic import (
    DetectorObservation,
    PhantomJamValidationStatus,
    detect_phantom_jam,
)


class FinalValidationV2Tests(unittest.TestCase):
    @staticmethod
    def _wave_observations(onsets):
        observations = []
        for position, onset in onsets:
            for time in range(65):
                active = onset <= time <= onset + 12
                observations.append(
                    DetectorObservation(
                        time=float(time),
                        position=float(position),
                        density=48.0 if active else 10.0,
                        speed=4.0 + 4.0 * (time % 2) if active else 20.0,
                    )
                )
        return observations

    def test_physically_implausible_wave_is_rejected(self):
        events = detect_phantom_jam(
            self._wave_observations(((0, 4), (100, 2), (200, 0))),
            30.0,
            9.0,
            8.0,
            3.0,
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(
            events[0].validation.status,
            PhantomJamValidationStatus.PHYSICALLY_IMPLAUSIBLE,
        )
        self.assertFalse(events[0].is_valid)

    def test_physically_plausible_wave_records_si_and_traffic_units(self):
        events = detect_phantom_jam(
            self._wave_observations(((0, 40), (100, 20), (200, 0))),
            30.0,
            9.0,
            8.0,
            3.0,
        )
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0].is_valid)
        self.assertAlmostEqual(events[0].backward_wave_speed, -5.0, places=6)
        self.assertAlmostEqual(
            events[0].validation.propagation_speed_kilometers_per_hour,
            -18.0,
            places=6,
        )

    def test_fixed_point_convergence_residual(self):
        result = solve_fixed_point(
            [0.0], lambda value: 0.5 * (value + 1.0), relaxation=1.0, tolerance=1e-8
        )
        self.assertTrue(result.converged)
        self.assertLess(result.final_residual, 1e-8)
        self.assertAlmostEqual(result.value[0], 1.0, places=6)

    def test_fixed_point_nonconvergence_is_explicit(self):
        result = solve_fixed_point(
            [0.0], lambda value: 1.0 - value, relaxation=1.0, max_iterations=6
        )
        self.assertFalse(result.converged)
        self.assertEqual(result.iterations, 6)
        self.assertGreater(result.final_residual, 0.9)

    def test_alignment_cost_is_monotone_in_epsilon(self):
        network, od, demand = two_route()
        routes = {
            route.route_id: route
            for route in network.multiobjective_candidate_routes(
                *od, k_per_objective=2, max_overlap=1.0, pareto_filter=False
            )
        }
        users = generate_population(4, *od, "high", 0.0, 5.0, 17)
        candidates = {user.user_id: tuple(routes) for user in users}
        frontier = compute_alignment_frontier(
            network,
            routes,
            users,
            candidates,
            demand / len(users),
            [0.0, 0.02, 0.08, 0.16, 0.24],
        )
        costs = [point.minimum_feasible_ttt for point in frontier.points]
        self.assertTrue(frontier.monotonic)
        self.assertTrue(all(right <= left + 1e-8 for left, right in zip(costs, costs[1:])))

    def test_real_topology_recommended_paths_are_legal(self):
        summary_path = (
            Path(__file__).resolve().parents[1]
            / "artifacts"
            / "studies"
            / "real_topology_policy_matrix"
            / "summary.json"
        )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertTrue(summary["all_recommended_paths_legal"])

    def test_rl0_actions_are_masked_eligibility_choices(self):
        policy = PPOEligibilityPolicy(3, 3, seed=7)
        states = np.asarray([[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]])
        mask = np.asarray([[True, False, True], [True, True, False]])
        probabilities = policy.probabilities(states, mask)
        self.assertTrue(np.allclose(probabilities.sum(axis=1), 1.0))
        self.assertEqual(probabilities[0, 1], 0.0)
        self.assertEqual(probabilities[1, 2], 0.0)


if __name__ == "__main__":
    unittest.main()
