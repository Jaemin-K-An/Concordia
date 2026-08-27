from __future__ import annotations

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import yaml

from concordia.v10.cache import RolloutCache
from concordia.v10.integrity import (
    assert_file_hashes,
    canonical_json_sha256,
    file_hashes,
    first_primes_at_or_above,
)
from concordia.v10.racing import MultiFidelityRacer, RolloutResult
from concordia.v10.seeds import assert_seed_contract, racing_seed
from concordia.v9.action_space import generate_action_library


ROOT = Path(__file__).resolve().parents[1]


class V10RacingTests(unittest.TestCase):
    def setUp(self):
        self.config = yaml.safe_load((ROOT / "configs/v10/racing_design.yaml").read_text())
        self.actions = [action.to_dict() for action in generate_action_library()[:7]]
        self.plans = {
            action["action_id"]: {
                "expected_accepted_user_count": 2.0,
                "destination_capacity_slack": 0.8,
                "preference_feasible": True,
                "all_routes_legal": True,
                "route_mapping_valid": True,
            }
            for action in self.actions
        }

    @staticmethod
    def evaluator(requests):
        actions = {action.action_id: action for action in generate_action_library()}
        output = []
        for request in requests:
            intensity = actions[request.action_id].reroute_fraction
            gain = 0.03 - 1.8 * (intensity - 0.10) ** 2
            risk = 0.40 if intensity >= 0.30 else 0.0
            output.append(RolloutResult(
                request.state_id, request.action_id, request.stage,
                request.horizon_seconds, request.replica, request.seed,
                gain, -gain, risk, -gain, 0.02, True,
            ))
        return output

    def test_racing_fixture_selects_interior_action(self):
        result = MultiFidelityRacer(self.config).race(
            "fixture", self.actions, self.plans, self.evaluator, evaluation_seed=811
        )
        selected = next(
            action for action in self.actions
            if action["action_id"] == result["selected_action_id"]
        )
        self.assertTrue(result["intervene"])
        self.assertEqual(selected["reroute_fraction"], 0.10)

    def test_null_action_always_exists(self):
        action_ids = {action["action_id"] for action in self.actions}
        self.assertIn("A00_NULL_B1", action_ids)

    def test_preference_violating_action_is_never_raced(self):
        self.plans["A01"]["preference_feasible"] = False
        result = MultiFidelityRacer(self.config).race(
            "fixture-preference", self.actions, self.plans, self.evaluator,
            evaluation_seed=814,
        )
        self.assertEqual(
            result["eliminated"]["A01"]["reason"],
            "preference_slack_violation",
        )
        self.assertFalse(any(
            request["action_id"] == "A01" for request in result["rollout_requests"]
        ))

    def test_eliminated_action_cannot_return(self):
        self.plans["A01"]["expected_accepted_user_count"] = 0.0
        result = MultiFidelityRacer(self.config).race(
            "fixture-filter", self.actions, self.plans, self.evaluator, evaluation_seed=812
        )
        self.assertEqual(result["eliminated"]["A01"]["stage"], "stage_0")
        for stage in ("stage_1_survivors", "stage_2_survivors", "stage_3_survivors"):
            self.assertNotIn("A01", result[stage])

    def test_seed_contract_and_cache(self):
        first = racing_seed("s1", "A01", "stage_1", 0)
        second = racing_seed("s1", "A01", "stage_2", 0)
        self.assertNotEqual(first, second)
        assert_seed_contract([first, second], [811])
        with self.assertRaises(RuntimeError):
            assert_seed_contract([first], [first])
        with tempfile.TemporaryDirectory() as temporary:
            cache = RolloutCache(Path(temporary))
            key = {
                "state_id": "s1", "action_id": "A01", "stage": "stage_1",
                "replica": 0, "horizon_seconds": 60,
                "simulator_parameter_hash": "abc",
            }
            cache.store(key, {"traffic_gain": 0.01})
            self.assertEqual(cache.load(key)["traffic_gain"], 0.01)

    def test_final_seed_commitment_and_freeze_hash_immutability(self):
        seeds = first_primes_at_or_above(5003, 25)
        self.assertEqual(
            canonical_json_sha256(seeds),
            "9ada7f84b093e34a69611dc6c4162dd6f879fd37dc8a204fc986deee48ea3b82",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frozen = root / "frozen.yaml"
            frozen.write_text("threshold: 0.005\n")
            expected = file_hashes([frozen], root)
            assert_file_hashes(expected, root)
            frozen.write_text("threshold: 0.004\n")
            with self.assertRaises(RuntimeError):
                assert_file_hashes(expected, root)

    def test_final_ids_are_absent_from_development_artifacts(self):
        seeds = first_primes_at_or_above(5003, 25)
        conditions = yaml.safe_load(
            (ROOT / "configs/v9/development_design.yaml").read_text()
        )["condition_templates"]
        final_ids = {
            f"V10F::{condition['id']}::s{seed}"
            for seed in seeds
            for condition in conditions
        }
        development_directory = ROOT / "artifacts/studies/v10_racing_validation"
        for path in development_directory.glob("*"):
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            self.assertFalse(
                "V10F::" in text,
                f"final state leaked into {path.relative_to(ROOT)}",
            )
        self.assertEqual(len(final_ids), 500)

    def test_unsafe_and_high_regret_finalists_are_rejected(self):
        def unsafe_evaluator(requests):
            return [RolloutResult(
                request.state_id, request.action_id, request.stage,
                request.horizon_seconds, request.replica, request.seed,
                0.10, 0.0, 0.30 if request.stage == "stage_3" else 0.0,
                0.0, 0.09 if request.stage == "stage_3" else 0.0, True,
            ) for request in requests]

        result = MultiFidelityRacer(self.config).race(
            "fixture-unsafe", self.actions, self.plans, unsafe_evaluator,
            evaluation_seed=813,
        )
        self.assertFalse(result["intervene"])
        self.assertEqual(result["selected_action_id"], "A00_NULL_B1")

    def test_stronger_fresh_verification_rejects_zero_gain_replica(self):
        config = deepcopy(self.config)
        config["verification"]["minimum_replica_relative_benefit"] = 0.005

        def reversal_evaluator(requests):
            return [RolloutResult(
                request.state_id, request.action_id, request.stage,
                request.horizon_seconds, request.replica, request.seed,
                0.0 if request.stage == "verification" and request.replica == 0 else 0.03,
                0.0, 0.0, 0.0, 0.02, True,
            ) for request in requests]

        result = MultiFidelityRacer(config).race(
            "fixture-reversal", self.actions, self.plans, reversal_evaluator,
            evaluation_seed=815,
        )
        self.assertFalse(result["intervene"])


if __name__ == "__main__":
    unittest.main()
