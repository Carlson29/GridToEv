import json
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed" / "gridtoev_model_ready.csv"
QUALITY = ROOT / "data" / "processed" / "gridtoev_quality_report.json"


class ProcessedDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = pd.read_csv(
            DATASET,
            parse_dates=["issue_timestamp_utc", "target_timestamp_utc"],
        )
        cls.quality = json.loads(QUALITY.read_text(encoding="utf-8"))

    def test_natural_key_is_unique(self) -> None:
        duplicate_count = self.data.duplicated(
            ["issue_timestamp_utc", "forecast_horizon_minutes"]
        ).sum()
        self.assertEqual(int(duplicate_count), 0)

    def test_dataset_has_no_missing_values(self) -> None:
        self.assertEqual(int(self.data.isna().sum().sum()), 0)

    def test_horizons_and_timestamps_match(self) -> None:
        self.assertEqual(set(self.data["forecast_horizon_minutes"]), {30, 60})
        actual_delta = self.data["target_timestamp_utc"] - self.data["issue_timestamp_utc"]
        expected_delta = pd.to_timedelta(self.data["forecast_horizon_minutes"], unit="m")
        self.assertTrue((actual_delta == expected_delta).all())

    def test_dispatch_event_matches_regression_target(self) -> None:
        expected = (self.data["dispatch_down_mwh"] > 0).astype("int8")
        self.assertTrue((self.data["dispatch_down_event"] == expected).all())

    def test_dispatch_down_accounting(self) -> None:
        difference = (
            self.data["dispatch_down_mwh"]
            - self.data["curtailment_mwh"]
            - self.data["constraint_mwh"]
        ).abs()
        self.assertLess(float(difference.max()), 0.05)

    def test_quality_report_matches_dataset(self) -> None:
        self.assertEqual(self.quality["rows"], len(self.data))
        self.assertEqual(self.quality["columns"], self.data.shape[1])


if __name__ == "__main__":
    unittest.main()

