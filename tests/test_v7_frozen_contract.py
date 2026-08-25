from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


class V7FrozenContractTest(unittest.TestCase):
    def test_frozen_hashes_are_immutable_when_manifest_exists(self):
        from v7_frozen import MANIFEST, verify_frozen

        if not MANIFEST.is_file():
            self.skipTest("v7 freeze has not been materialized")
        before = verify_frozen()["manifest_self_hash"]
        after = verify_frozen()["manifest_self_hash"]
        self.assertEqual(before, after)

    def test_final_ids_are_absent_from_all_fitting_when_final_exists(self):
        final_path = ROOT / "artifacts/studies/v7_frozen_micro_holdout/raw_metrics.json"
        manifest_path = ROOT / "artifacts/studies/v7_model_selection/training_manifest.json"
        if not final_path.is_file():
            self.skipTest("v7 final holdout has not been materialized")
        final_ids = {row["pair_id"] for row in json.loads(final_path.read_text())}
        manifest = json.loads(manifest_path.read_text())
        training_ids = {
            case_id for values in manifest["case_ids"].values() for case_id in values
        }
        self.assertFalse(final_ids & training_ids)


if __name__ == "__main__":
    unittest.main()
