import random
import unittest

from concordia.behavior import (
    AcceptanceModel,
    AcceptanceOutcome,
    RecommendationDecision,
    RouteOffer,
)
from concordia.simulation import SumoAdapter


def make_offer(probability=1.0):
    return RouteOffer(
        offer_id="offer-1",
        user_id="vehicle-1",
        current_route_id="current",
        candidate_route_id="candidate",
        executable_edge_ids=("e0", "e1"),
        expected_eta_minutes=9.0,
        eta_variance_minutes2=1.0,
        monetary_cost=0.0,
        safety_risk=0.1,
        complexity=0.2,
        familiarity=0.0,
        estimated_utility=-0.2,
        reference_utility=-0.3,
        preference_slack=0.0,
        network_marginal_benefit_vehicle_minutes=2.0,
        predicted_acceptance_probability=probability,
        timestamp_seconds=10.0,
        model_version="acceptance-v1",
        coefficient_source="synthetic_assumption",
    )


class FakeEdgeAPI:
    def getIDList(self):
        return ("e0",)

    def getLastStepVehicleNumber(self, edge_id):
        return 10

    def getLaneNumber(self, edge_id):
        return 2

    def getLength(self, edge_id):
        return 500.0

    def getLastStepMeanSpeed(self, edge_id):
        return 10.0

    def getLastStepOccupancy(self, edge_id):
        return 12.5


class FakeSimulationAPI:
    def getTime(self):
        return 42.0


class FakeLaneAPI:
    def getLength(self, lane_id):
        return 500.0


class FakeVehicleAPI:
    def __init__(self):
        self.calls = []

    def setRoute(self, vehicle_id, edges):
        self.calls.append((vehicle_id, edges))


class FakeTraci:
    def __init__(self):
        self.edge = FakeEdgeAPI()
        self.lane = FakeLaneAPI()
        self.simulation = FakeSimulationAPI()
        self.vehicle = FakeVehicleAPI()

    def simulationStep(self):
        return None


class BehaviorAndSumoStateTests(unittest.TestCase):
    def test_count_density_flow_speed_and_occupancy_have_distinct_units(self):
        adapter = SumoAdapter("unused.sumocfg")
        adapter._traci = FakeTraci()
        observation = adapter.step().edges["e0"]
        self.assertEqual(observation.vehicle_count, 10)
        self.assertEqual(observation.density_vehicles_per_km_per_lane, 10.0)
        self.assertEqual(observation.mean_speed_meters_per_second, 10.0)
        self.assertEqual(observation.flow_vehicles_per_hour_per_lane, 360.0)
        self.assertEqual(observation.occupancy_percent, 12.5)

    def test_rejected_offer_never_calls_set_route(self):
        adapter = SumoAdapter("unused.sumocfg")
        adapter._traci = FakeTraci()
        decision = RecommendationDecision(
            offer=make_offer(0.0),
            outcome=AcceptanceOutcome.REJECTED,
            sampled_probability=0.5,
            decided_at_seconds=10.0,
            reason="user_rejected_offer",
        )
        self.assertFalse(adapter.execute_accepted_route(decision))
        self.assertEqual(adapter._traci.vehicle.calls, [])

    def test_only_accepted_offer_executes_candidate_edges(self):
        adapter = SumoAdapter("unused.sumocfg")
        adapter._traci = FakeTraci()
        decision = AcceptanceModel().decide(make_offer(1.0), random.Random(7), 10.0)
        self.assertTrue(decision.accepted)
        self.assertTrue(adapter.execute_accepted_route(decision))
        self.assertEqual(adapter._traci.vehicle.calls, [("vehicle-1", ["e0", "e1"])])

    def test_acceptance_probability_decreases_with_slack(self):
        model = AcceptanceModel()
        low = model.probability(0.0, 0.0, 0.0, 0.0, 0.0)
        high = model.probability(0.3, 0.0, 0.0, 0.0, 0.0)
        self.assertGreater(low, high)


if __name__ == "__main__":
    unittest.main()
