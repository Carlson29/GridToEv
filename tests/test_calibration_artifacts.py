import hashlib
import json
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "benchmarks" / "calibrated_intervals"


class CalibrationArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads((OUTPUT / "calibration_report.json").read_text(encoding="utf-8"))
        cls.policy = json.loads((OUTPUT / "interval_policy.json").read_text(encoding="utf-8"))
        cls.comparison = pd.read_csv(OUTPUT / "development_interval_comparison.csv")

    def test_policy_is_derived_from_issue_8_oof_only(self) -> None:
        oof = ROOT / "benchmarks" / "horizon_models" / "development_oof_predictions.csv.gz"
        self.assertEqual(self.report["oof_sha256"], hashlib.sha256(oof.read_bytes()).hexdigest())
        self.assertEqual(self.policy["calibration_rows"], 1634)
        self.assertEqual(set(self.policy["offsets_by_horizon"]), {"30", "60"})
        self.assertFalse(self.report["selection"]["final_test_labels_used"])
        self.assertTrue(self.report["selection"]["selected_before_final_test_scoring"])
        self.assertTrue(self.report["final_test"]["accessed_after_selection"])

    def test_comparison_and_final_decision_are_honest(self) -> None:
        self.assertEqual(self.comparison["method"].nunique(), 4)
        self.assertEqual(self.comparison["score_fold"].nunique(), 3)
        self.assertGreater(self.report["final_test"]["candidate"]["average_width_mwh"], 0)
        self.assertFalse(self.report["release_candidate_eligible"])
        self.assertFalse(self.report["release_checks"]["coverage_min"])
        self.assertEqual(self.report["production_decision"], "retain_v1.1.0_intervals")
        self.assertFalse(self.report["production_bundle_modified"])

    def test_components_reconcile_without_refit(self) -> None:
        components = self.report["components"]
        self.assertFalse(components["refitted"])
        self.assertFalse(components["refit_eligibility"]["eligible"])
        self.assertLessEqual(components["reconciliation_max_absolute_error_mwh"], 1e-9)


if __name__ == "__main__":
    unittest.main()
