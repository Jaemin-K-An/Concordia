import unittest

from concordia.errors import ValidationError
from concordia.models import EdgeData, Route, RouteFeatures
from concordia.network import RoadNetwork
from concordia.scenarios import two_route


class NetworkTests(unittest.TestCase):
    def test_candidate_paths_are_connected_and_distinct(self):
        network, od, _ = two_route()
        paths = network.candidate_paths(*od, k=2, max_overlap=0.5)
        self.assertEqual(len(paths), 2)
        for path in paths:
            self.assertEqual(path[0], od[0])
            self.assertEqual(path[-1], od[1])
            network.path_edges(path)
        self.assertEqual(network.overlap_coefficient(paths[0], paths[1]), 0.0)

    def test_illegal_edges_are_never_candidates(self):
        network = RoadNetwork()
        network.add_edge("O", "X", EdgeData(1, 100, legal=False))
        network.add_edge("X", "D", EdgeData(1, 100))
        network.add_edge("O", "D", EdgeData(5, 100))
        self.assertEqual(network.candidate_paths("O", "D", k=1), [("O", "D")])
        with self.assertRaises(ValidationError):
            network.path_edges(("O", "X", "D"))

    def test_pareto_filter_removes_dominated_route(self):
        better = Route("better", ("O", "A", "D"), RouteFeatures(10, risk=0.1))
        worse = Route("worse", ("O", "B", "D"), RouteFeatures(12, risk=0.2))
        front = RoadNetwork.pareto_front([better, worse])
        self.assertEqual([route.route_id for route in front], ["better"])


if __name__ == "__main__":
    unittest.main()
