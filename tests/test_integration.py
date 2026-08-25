import json
import tempfile
import unittest
from pathlib import Path

from concordia.config import load_config
from concordia.evaluation import ExperimentRegistry
from concordia.experiment import run_experiment
from concordia.gis import audit_topology, export_edge_geojson, import_osm_xml
from concordia.populations import generate_population
from concordia.reporting import build_report
from concordia.safety import parse_sumo_ssm, summarize_ssm_conflict_types
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
            self.assertEqual(manifest["status"], "valid")
            self.assertIn("git_dirty", manifest)
            self.assertIn("end_timestamp_utc", manifest)
            self.assertIn("solver_version", manifest)
            self.assertIn("output_hashes", manifest)
            self.assertEqual(
                len((run_dir / "decisions.jsonl").read_text(encoding="utf-8").splitlines()),
                len(config["seeds"]) * config["user_count"],
            )
            report = build_report(str(runs), str(Path(temp) / "report.html"))
            self.assertIn("Synthetic analytical results", report.read_text(encoding="utf-8"))

    def test_registry_marks_incomplete_and_nonfinite_runs_invalid(self):
        with tempfile.TemporaryDirectory() as temp:
            config = {"seeds": [7], "name": "invalid-fixture"}
            run = ExperimentRegistry(temp).create(
                config,
                {"complete": False, "metric": float("nan")},
            )
            manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
            metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "invalid")
            self.assertTrue(manifest["invalid_reasons"])
            self.assertIsNone(metrics["metric"])

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
                '<ssm><conflict begin="1" end="2" ego="e" foe="f" type="merge">'
                '<minTTC value="1.2"/><minPET value="0.7"/><maxDRAC value="3.4"/>'
                '</conflict></ssm>',
                encoding="utf-8",
            )
            conflicts = parse_sumo_ssm(str(ssm))
            self.assertEqual(len(conflicts), 1)
            self.assertEqual(conflicts[0].min_ttc, 1.2)
            self.assertEqual(conflicts[0].max_mdrac, None)
            self.assertEqual(conflicts[0].conflict_type, "merge")
            self.assertEqual(summarize_ssm_conflict_types(conflicts)["merge"], 1)

    def test_osm_import_preserves_provenance_and_real_geometry(self):
        osm_xml = """<osm version="0.6">
          <node id="1" lat="37.0" lon="127.0"/>
          <node id="2" lat="37.001" lon="127.001"/>
          <node id="3" lat="37.002" lon="127.002"/>
          <node id="4" lat="37.001" lon="127.003"/>
          <way id="10"><nd ref="1"/><nd ref="2"/><nd ref="3"/>
            <tag k="highway" v="primary"/><tag k="lanes" v="2"/></way>
          <way id="11"><nd ref="1"/><nd ref="4"/><nd ref="3"/>
            <tag k="highway" v="secondary"/></way>
        </osm>"""
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "network.osm"
            source.write_text(osm_xml, encoding="utf-8")
            imported = import_osm_xml(
                str(source),
                source_url="https://api.openstreetmap.org/test",
                retrieval_date="2026-08-25",
            )
            audit = audit_topology(imported.network, "1", "3")
            self.assertTrue(audit.valid)
            self.assertEqual(audit.alternative_route_count, 2)
            self.assertEqual(imported.provenance.demand_provenance, "synthetic demand on real topology")
            geometry = dict(imported.geometries)
            geometry[("1", "2")] = ((127.0, 37.0), (127.0004, 37.0007), (127.001, 37.001))
            layer = export_edge_geojson(
                imported.network,
                imported.coordinates,
                str(Path(temp) / "real.geojson"),
                geometries=geometry,
                provenance_source=imported.provenance.source_url,
            )
            payload = json.loads(layer.read_text(encoding="utf-8"))
            feature = next(item for item in payload["features"] if item["id"] == "1->2")
            self.assertEqual(len(feature["geometry"]["coordinates"]), 3)


if __name__ == "__main__":
    unittest.main()
