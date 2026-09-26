import tempfile
import unittest
from pathlib import Path

import pandas as pd

from gridtoev.actuals import ActualsService
from gridtoev.constants import DEFAULT_DATASET_PATH
from gridtoev.daily_curtailment import DEFAULT_DAILY_DATASET, DEFAULT_HISTORY


class ActualsServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self.temp_dir.name) / "history.csv.gz"
        hours = pd.date_range("2026-01-01", periods=48, freq="30min", tz="UTC")
        self.frame = pd.DataFrame({
            "timestamp_utc": hours,
            "dispatch_down_mwh": [3.0] * 48,
            "curtailment_mwh": [2.0] * 48,
            "constraint_mwh": [1.0] * 48,
            "dispatch_down_available_flag": [1] * 48,
        })
        self.frame.to_csv(self.path, index=False, compression="gzip")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_half_hour_actual_uses_target_not_issue_time(self):
        actuals = ActualsService(self.path)
        result = actuals.point(pd.Timestamp("2026-01-01T00:30:00Z"))
        self.assertEqual(result["status"], "available")
        self.assertEqual(result["actual_dispatch_down_mwh"], 3.0)
        self.assertEqual(result["actual_curtailment_mwh"], 2.0)
        self.assertEqual(result["actual_constraint_mwh"], 1.0)
        self.assertEqual(result["target_timestamp_utc"], "2026-01-01T00:30:00+00:00")
        self.assertEqual(actuals.point(pd.Timestamp("2026-01-02T00:00:00Z"))["status"], "pending")

    def test_daily_actual_requires_complete_day(self):
        actuals = ActualsService(self.path)
        available = actuals.day(pd.Timestamp("2026-01-01").date())
        self.assertEqual(available["status"], "available")
        self.assertEqual(available["actual_curtailment_mwh"], 96.0)
        self.assertTrue(available["actual_curtailment_event"])
        self.assertEqual(actuals.day(pd.Timestamp("2026-01-02").date())["status"], "pending")

    def test_internal_gap_and_invalid_label_are_not_actuals(self):
        late = self.frame.iloc[[-1]].copy()
        late["timestamp_utc"] = pd.Timestamp("2026-01-03T00:00:00Z")
        frame = pd.concat([self.frame.iloc[:-1], late], ignore_index=True)
        frame.loc[0, "dispatch_down_available_flag"] = 0
        frame.to_csv(self.path, index=False, compression="gzip")
        actuals = ActualsService(self.path)
        self.assertEqual(actuals.point(pd.Timestamp("2026-01-01T00:00:00Z"))["status"], "missing")
        self.assertEqual(actuals.point(pd.Timestamp("2026-01-01T23:30:00Z"))["status"], "missing")
        self.assertEqual(actuals.day(pd.Timestamp("2026-01-01").date())["status"], "missing")
        self.assertEqual(actuals.point(pd.Timestamp("2026-01-04T00:00:00Z"))["status"], "pending")

    def test_batch_preserves_order_and_rejects_naive_or_unaligned_times(self):
        actuals = ActualsService(self.path)
        times = [pd.Timestamp("2026-01-02T00:00:00Z"), pd.Timestamp("2026-01-01T00:30:00Z")]
        self.assertEqual([row["status"] for row in actuals.points(times)], ["pending", "available"])
        with self.assertRaisesRegex(ValueError, "timezone"):
            actuals.point(pd.Timestamp("2026-01-01T00:30:00"))
        with self.assertRaisesRegex(ValueError, "half-hour"):
            actuals.point(pd.Timestamp("2026-01-01T00:15:00Z"))


class CommittedActualsAlignmentTests(unittest.TestCase):
    def test_v1_target_timestamps_match_eirgrid_labels(self):
        actuals = ActualsService(DEFAULT_HISTORY)
        model_rows = pd.read_csv(DEFAULT_DATASET_PATH).iloc[::97]
        for _, row in model_rows.iterrows():
            observed = actuals.point(pd.Timestamp(row["target_timestamp_utc"]))
            self.assertEqual(observed["status"], "available")
            self.assertAlmostEqual(observed["actual_dispatch_down_mwh"], row["dispatch_down_mwh"])
            self.assertAlmostEqual(observed["actual_curtailment_mwh"], row["curtailment_mwh"])
            self.assertAlmostEqual(observed["actual_constraint_mwh"], row["constraint_mwh"])

    def test_daily_target_dates_match_eirgrid_label_sums(self):
        actuals = ActualsService(DEFAULT_HISTORY)
        model_rows = pd.read_csv(DEFAULT_DAILY_DATASET).iloc[::71]
        for _, row in model_rows.iterrows():
            observed = actuals.day(pd.Timestamp(row["issue_timestamp_utc"]).date())
            self.assertEqual(observed["status"], "available")
            self.assertAlmostEqual(observed["actual_curtailment_mwh"], row["curtailment_mwh"])


if __name__ == "__main__":
    unittest.main()
