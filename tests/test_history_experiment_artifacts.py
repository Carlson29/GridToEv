import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gridtoev.history_model import source_sha256


class HistoryExperimentArtifactTests(unittest.TestCase):
    def test_predeclared_experiment_preserves_final_holdout(self) -> None:
        contract = json.loads((ROOT / "config" / "multi_year_causal_experiment.v1.json").read_text(encoding="utf-8"))
        report = json.loads((ROOT / "benchmarks" / "multi_year_causal" / "experiment_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["experiment_id"], contract["experiment_id"])
        self.assertEqual(report["source_sha256"], source_sha256(ROOT / contract["source_path"]))
        self.assertEqual(report["source_rows"], 99310)
        self.assertEqual(len(report["development"]["folds"]), 3)
        self.assertTrue(all(fold["label_maturity_verified"] for fold in report["development"]["folds"]))
        self.assertFalse(report["development"]["passed"])
        self.assertFalse(report["sealed_final"]["accessed"])
        self.assertIsNone(report["sealed_final"]["score"])
        self.assertFalse(report["release"]["candidate_approved"])


if __name__ == "__main__":
    unittest.main()
