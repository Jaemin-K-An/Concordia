import json
import tempfile
import unittest
from pathlib import Path

from concordia.config import load_config
from concordia.evaluation import ExperimentRegistry
from concordia.experiment import run_experiment
from concordia.gis import export_edge_geojson
from concordia.populations import generate_population
from concordia.reporting import build_report
from concordia.safety import parse_sumo_ssm
from concordia.scenarios import two_route
from concordia.simulation import SumoAdapter
from concordia.errors import SimulatorUnavailable


class IntegrationTests(unittest.TestCase):
    def test_registered_experiment_and_report(self):
        config = load_config("configs/experiments/smoke.yaml")
        metrics = run_experiment(config)
        self.assertTrue(metrics["ue"]["converged"])
        self.assertTrue(metrics["so"]["converged"])
        self.assertLessEqual(metrics["proposed_ttt"]["mean"], metrics["private_best_ttt"]["mean"])
        self.assertLessEqual(metrics["proposed_regret"]["max"], config["utility_epsilon"] + 1e-10)
        with tempfile.TemporaryDirectory() as temp:
            runs = Path(temp) / "runs"
            run_dir = ExperimentRegistry(str(runs)).create(config, metrics)
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["seeds"], config["seeds"])
            self.assertEqual(
                len((run_dir / "decisions.jsonl").read_text(encoding="utf-8").splitlines()),
                len(config["seeds"]) * config["user_count"],
            )
            report = build_report(str(runs), str(Path(temp) / "report.html"))
            self.assertIn("Synthetic analytical results", report.read_text(encoding="utf-8"))

    def test_sumo_missing_fails_explicitly(self):
        adapter = SumoAdapter("missing.sumocfg", binary="certainly-not-a-sumo-binary")
        with self.assertRaises(SimulatorUnavailable):
            adapter.start(seed=42)

    def test_zero_penetration_preserves_private_assignment(self):
        config = load_config("configs/experiments/smoke.yaml")
        config["navigation_penetration"] = 0.0
        metrics = run_experiment(config)
        self.assertAlmostEqual(
            metrics["proposed_ttt"]["mean"], metrics["private_best_ttt"]["mean"]
        )

    def test_matched_population_mean_and_variance(self):
        kwargs = dict(count=20, origin="O", destination="D", epsilon=0.1, rationality=5, seed=7)
        low = generate_population(heterogeneity="low", **kwargs)
        high = generate_population(heterogeneity="high", **kwargs)
        low_weights = [[getattr(user.preferences, name) for name in ("time", "variability", "cost", "risk", "complexity", "familiarity")] for user in low]
        high_weights = [[getattr(user.preferences, name) for name in ("time", "variability", "cost", "risk", "complexity", "familiarity")] for user in high]
        import numpy as np

        self.assertTrue(np.allclose(np.mean(low_weights, axis=0), np.mean(high_weights, axis=0)))
        self.assertGreater(np.var(high_weights), np.var(low_weights))

    def test_qgis_export_and_ssm_parser(self):
        network, _, _ = two_route()
        coordinates = {"O": (127.0, 37.0), "A": (127.01, 37.01), "B": (127.01, 36.99), "D": (127.02, 37.0)}
        with tempfile.TemporaryDirectory() as temp:
            layer = export_edge_geojson(network, coordinates, str(Path(temp) / "edges.geojson"))
            self.assertTrue(layer.is_file())
            self.assertTrue(layer.with_suffix(".geojson.manifest.json").is_file())
            ssm = Path(temp) / "ssm.xml"
            ssm.write_text(
                '<ssm><conflict begin="1" end="2" ego="e" foe="f">'
                '<minTTC value="1.2"/><minPET value="0.7"/><maxDRAC value="3.4"/>'
                '</conflict></ssm>',
                encoding="utf-8",
            )
            conflicts = parse_sumo_ssm(str(ssm))
            self.assertEqual(len(conflicts), 1)
            self.assertEqual(conflicts[0].min_ttc, 1.2)


if __name__ == "__main__":
    unittest.main()
