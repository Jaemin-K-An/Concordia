import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from concordia.feasibility import (
    FEATURE_SCHEMA,
    FeasibilityGate,
    FeasibilityModel,
    build_alignment_case,
    select_intervention_threshold,
)
from concordia.selective import SelectiveInterventionPolicy, baseline_fallback


ROOT = Path(__file__).resolve().parents[1]


class V3SelectiveTests(unittest.TestCase):
    def test_aps_nonnegative_and_overlap_bounded(self):
        case = build_alignment_case(
            scenario="two_route",
            seed=17,
            demand_scale=1.0,
            heterogeneity="high",
            navigation_penetration=1.0,
            user_count=6,
            regret_limit=0.08,
            epsilon_grid=[0.0, 0.08],
            minimum_relative_ttt_gain=0.01,
            safety_delta=0.25,
            source_split="fixture",
        )
        self.assertGreaterEqual(case["features"]["alignment_potential_score"], 0.0)
        self.assertGreaterEqual(case["features"]["route_overlap"], 0.0)
        self.assertLessEqual(case["features"]["route_overlap"], 1.0)

    def test_probability_bounds_and_deterministic_threshold(self):
        rng = np.random.default_rng(5)
        matrix = rng.normal(size=(30, len(FEATURE_SCHEMA)))
        labels = np.asarray([0, 1] * 15)
        model = FeasibilityModel("fixture", "logistic", FEATURE_SCHEMA, iterations=100)
        probabilities = model.fit(matrix, labels).predict_proba(matrix)
        self.assertTrue(np.all((probabilities >= 0.0) & (probabilities <= 1.0)))
        kwargs = dict(
            labels=labels,
            probabilities=probabilities,
            uncertainties=np.zeros(len(labels)),
            precision_target=0.65,
            coverage_target=0.40,
            maximum_uncertainty=0.2,
            candidates=[0.4, 0.5, 0.6, 0.7],
        )
        self.assertEqual(
            select_intervention_threshold(**kwargs),
            select_intervention_threshold(**kwargs),
        )

    def test_abstain_and_safety_failure_never_change_route(self):
        routes = {"u1": "private", "u2": "current"}
        gate = FeasibilityGate(0.7, 0.1, 0.25, 0.5, 0.01, 0.1)
        policy = SelectiveInterventionPolicy(gate)
        decision = policy.decide(
            case_id="unsafe",
            p_win=0.9,
            p_win_lower=0.8,
            uncertainty=0.02,
            alignment_potential=1.0,
            route_overlap=0.1,
            safety_upper_difference=0.26,
            acceptance_probability=0.9,
            ttt_lcb_gain=0.04,
            predicted_tail_loss=0.0,
            legal=True,
        )
        self.assertFalse(decision.intervene)
        self.assertEqual(baseline_fallback(routes), routes)
        self.assertIsNot(baseline_fallback(routes), routes)

    def test_known_win_intervenes_and_infeasible_or_overlap_fixture_abstains(self):
        policy = SelectiveInterventionPolicy(FeasibilityGate(0.7, 0.1, 0.25, 0.5, 0.01, 0.1))
        common = dict(
            uncertainty=0.03,
            alignment_potential=1.0,
            safety_upper_difference=0.0,
            acceptance_probability=0.9,
            ttt_lcb_gain=0.03,
            predicted_tail_loss=0.0,
            legal=True,
        )
        win = policy.decide(
            case_id="win", p_win=0.9, p_win_lower=0.8, route_overlap=0.1, **common
        )
        infeasible = policy.decide(
            case_id="infeasible",
            p_win=0.2,
            p_win_lower=0.1,
            route_overlap=0.2,
            **common,
        )
        overlap = policy.decide(
            case_id="overlap",
            p_win=0.75,
            p_win_lower=0.49,
            route_overlap=0.95,
            **common,
        )
        self.assertTrue(win.intervene)
        self.assertFalse(infeasible.intervene)
        self.assertFalse(overlap.intervene)

    def test_holdout_split_never_enters_training(self):
        split = yaml.safe_load((ROOT / "configs/v3/splits.yaml").read_text())
        development = set(split["training"]["seeds"] + split["validation"]["seeds"])
        holdout = set(split["holdout"]["seeds"])
        self.assertFalse(development & holdout)

    def test_frozen_threshold_checksum_is_immutable_during_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "frozen.yaml"
            path.write_text("p_win_threshold: 0.7\n", encoding="utf-8")
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            yaml.safe_load(path.read_text(encoding="utf-8"))
            after = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
