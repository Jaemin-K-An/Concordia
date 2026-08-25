import unittest

from concordia.scenarios import braess, two_route
import numpy as np

from concordia.traffic import (
    DetectorObservation,
    LogisticPhantomJamRiskPredictor,
    StumpEnsemblePhantomJamRiskPredictor,
    TrafficAssignment,
    calibration_metrics,
    detect_phantom_jam,
)


class TrafficAssignmentTests(unittest.TestCase):
    def test_ue_and_so_converge_and_poa_is_valid(self):
        network, od, demand = two_route()
        assignment = TrafficAssignment(network)
        ue = assignment.user_equilibrium({od: demand})
        so = assignment.system_optimum({od: demand})
        self.assertTrue(ue.converged)
        self.assertTrue(so.converged)
        self.assertLessEqual(so.total_travel_time, ue.total_travel_time + 1e-5)
        self.assertGreaterEqual(ue.total_travel_time / so.total_travel_time, 1.0)

    def test_flow_conservation(self):
        network, od, demand = two_route()
        result = TrafficAssignment(network).user_equilibrium({od: demand})
        for node in network.graph.nodes:
            inflow = sum(result.flows[(source, node)] for source in network.graph.predecessors(node))
            outflow = sum(result.flows[(node, target)] for target in network.graph.successors(node))
            expected = -demand if node == od[0] else demand if node == od[1] else 0.0
            self.assertAlmostEqual(inflow - outflow, expected, places=5)

    def test_braess_golden_regression(self):
        base, od, demand = braess(with_connector=False)
        connected, connected_od, _ = braess(with_connector=True)
        base_result = TrafficAssignment(base).user_equilibrium({od: demand})
        connected_result = TrafficAssignment(connected).user_equilibrium({connected_od: demand})
        self.assertTrue(base_result.converged and connected_result.converged)
        self.assertGreater(connected_result.total_travel_time, base_result.total_travel_time * 1.20)
        self.assertAlmostEqual(base_result.total_travel_time, 260040.0, delta=30.0)
        self.assertAlmostEqual(connected_result.total_travel_time, 320120.0, delta=30.0)

    def test_phantom_jam_requires_backward_sustained_wave(self):
        observations = []
        # Downstream detector (x=100) collapses first; the onset later reaches x=0.
        for position, onset in ((0.0, 12.0), (100.0, 10.0), (200.0, 8.0)):
            for time in range(0, 20):
                active = onset <= time <= onset + 4
                speed = 4.0 + (time % 2) * 4.0 if active else 20.0
                density = 45.0 if active else 10.0
                observations.append(DetectorObservation(time, position, density, speed))
        events = detect_phantom_jam(
            observations,
            critical_density=30,
            low_speed_threshold=9,
            minimum_duration=3,
            minimum_amplitude=3,
        )
        self.assertEqual(len(events), 1)
        self.assertLess(events[0].backward_wave_speed, 0)
        self.assertEqual(events[0].detector_count, 3)

    def test_speed_drop_without_propagation_is_not_phantom_jam(self):
        observations = [
            DetectorObservation(time, 0, 40, 5 if 2 <= time <= 8 else 20)
            for time in range(12)
        ]
        self.assertEqual(detect_phantom_jam(observations, 30, 10, 3, 2), [])

    def test_detector_tracks_multiple_separated_events(self):
        observations = []
        for event_start in (10, 40):
            for position, offset in ((0.0, 4), (100.0, 2), (200.0, 0)):
                for time in range(event_start, event_start + 10):
                    active = event_start + offset <= time <= event_start + offset + 4
                    observations.append(
                        DetectorObservation(
                            time,
                            position,
                            45 if active else 10,
                            4 + (time % 2) * 4 if active else 20,
                        )
                    )
        events = detect_phantom_jam(observations, 30, 9, 3, 3)
        self.assertEqual(len(events), 2)
        self.assertTrue(all(event.affected_length == 200 for event in events))
        self.assertTrue(all(0 < event.confidence <= 1 for event in events))

    def test_phantom_predictor_calibration_metrics(self):
        rng = np.random.default_rng(4)
        features = rng.normal(size=(240, 8))
        labels = ((1.8 * features[:, 6] + features[:, 0] - features[:, 1]) > 0).astype(int)
        train_x, test_x = features[:180], features[180:]
        train_y, test_y = labels[:180], labels[180:]
        logistic = LogisticPhantomJamRiskPredictor(iterations=1200).fit(train_x, train_y)
        tree = StumpEnsemblePhantomJamRiskPredictor().fit(train_x, train_y)
        logistic_metrics = calibration_metrics(test_y, logistic.predict_proba(test_x))
        tree_metrics = calibration_metrics(test_y, tree.predict_proba(test_x))
        self.assertGreater(logistic_metrics.roc_auc, 0.85)
        self.assertGreater(tree_metrics.roc_auc, 0.65)
        self.assertIn("coefficients", logistic.model_card())


if __name__ == "__main__":
    unittest.main()
