import tempfile
import unittest
import hashlib
import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from gridtoev.daily_model import (
    DEFAULT_ARTIFACT, DEFAULT_REPORT, DailyCurtailmentService,
    _validate_dataset, chronological_partitions, feature_columns,
)
from gridtoev.daily_curtailment import DEFAULT_DAILY_DATASET


class DailyModelTests(unittest.TestCase):
    def test_forward_window_returns_seven_causal_daily_predictions(self):
        service = DailyCurtailmentService(DEFAULT_ARTIFACT, DEFAULT_REPORT)
        today = datetime.now(timezone.utc).date()
        start = today + timedelta(days=1)
        hours = pd.date_range(start, periods=7 * 24, freq="h", tz="UTC")
        retrieved = pd.Timestamp.now(tz="UTC")

        def live_region(region, first, last):
            self.assertEqual((first, last), (start, today + timedelta(days=7)))
            return pd.DataFrame({
                "target_hour_utc": hours, "region": region,
                "available_at_utc": retrieved,
                "wind_speed_100m_kmh": [20.0] * len(hours),
                "shortwave_radiation_wm2": [100.0] * len(hours),
                "temperature_2m_c": [12.0] * len(hours),
            })

        with patch("gridtoev.daily_model.fetch_live_forecast", side_effect=live_region) as fetch:
            result = service.predict_forward_window(start, 7)
        self.assertEqual(fetch.call_count, 4)
        self.assertEqual(result["prediction_count"], 7)
        self.assertEqual(len(result["predictions"]), 7)
        self.assertEqual(result["predictions"][0]["target_date_utc"], start.isoformat())
        self.assertEqual(result["predictions"][-1]["target_date_utc"], (today + timedelta(days=7)).isoformat())
        self.assertTrue(all(pd.Timestamp(item["issue_timestamp_utc"]) >= retrieved for item in result["predictions"]))
        self.assertTrue(all(item["forecast_lead_hours"] > 0 for item in result["predictions"]))
        self.assertIn("not validated", result["notice"].lower())

    def test_forward_window_rejects_invalid_range_before_fetch(self):
        service = DailyCurtailmentService(DEFAULT_ARTIFACT, DEFAULT_REPORT)
        today = datetime.now(timezone.utc).date()
        with patch("gridtoev.daily_model.fetch_live_forecast") as fetch:
            with self.assertRaisesRegex(ValueError, "tomorrow"):
                service.predict_forward_window(today, 1)
            with self.assertRaisesRegex(ValueError, "seven"):
                service.predict_forward_window(today + timedelta(days=7), 2)
            fetch.assert_not_called()

    def test_forward_window_rejects_missing_hour_without_partial_results(self):
        service = DailyCurtailmentService(DEFAULT_ARTIFACT, DEFAULT_REPORT)
        today = datetime.now(timezone.utc).date()
        start = today + timedelta(days=1)
        hours = pd.date_range(start, periods=24, freq="h", tz="UTC")[:-1]

        def incomplete(region, first, last):
            return pd.DataFrame({
                "target_hour_utc": hours, "region": region,
                "available_at_utc": pd.Timestamp.now(tz="UTC"),
                "wind_speed_100m_kmh": [20.0] * len(hours),
                "shortwave_radiation_wm2": [100.0] * len(hours),
                "temperature_2m_c": [12.0] * len(hours),
            })

        with patch("gridtoev.daily_model.fetch_live_forecast", side_effect=incomplete):
            with self.assertRaisesRegex(OSError, "complete"):
                service.predict_forward_window(start, 1)

    def test_partitions_are_strictly_chronological(self):
        dates = pd.to_datetime(["2024-12-31", "2025-01-01", "2025-12-31", "2026-01-01"], utc=True)
        frame = pd.DataFrame({"issue_timestamp_utc": dates})
        train, validation, test = chronological_partitions(frame)
        self.assertEqual((len(train), len(validation), len(test)), (1, 2, 1))

    def test_model_features_exclude_labels_and_provenance(self):
        frame = pd.DataFrame({
            "west_wind_mean_kmh": [1.0],
            "calendar_day_sin": [0.0],
            "curtailment_mwh": [3.0],
            "curtailment_event": [1],
            "forecast_max_available_at_utc": ["2025-01-01T00:00:00Z"],
        })
        self.assertEqual(feature_columns(frame), ["west_wind_mean_kmh", "calendar_day_sin"])

    def test_service_rejects_unavailable_future_issue_date(self):
        with tempfile.TemporaryDirectory() as directory:
            service = DailyCurtailmentService(Path(directory) / "missing.joblib")
            with self.assertRaisesRegex(ValueError, "future"):
                service.predict_date(pd.Timestamp.now(tz="UTC").date() + pd.Timedelta(days=1))
            with self.assertRaisesRegex(ValueError, "before 2024-04-01"):
                service.predict_date(pd.Timestamp("2024-03-31").date())

    def test_dataset_validation_rejects_post_issue_forecast(self):
        frame = pd.read_csv(DEFAULT_DAILY_DATASET, nrows=1)
        frame.loc[0, "forecast_max_available_at_utc"] = "2025-01-01T01:00:00Z"
        with self.assertRaisesRegex(ValueError, "after prediction issue"):
            _validate_dataset(frame)

    def test_committed_artifact_and_dataset_match_report(self):
        report = json.loads(DEFAULT_REPORT.read_text(encoding="utf-8"))
        self.assertEqual(hashlib.sha256(DEFAULT_ARTIFACT.read_bytes()).hexdigest(), report["artifact_sha256"])
        self.assertEqual(hashlib.sha256(DEFAULT_DAILY_DATASET.read_bytes()).hexdigest(), report["dataset_sha256"])
        frame = pd.read_csv(DEFAULT_DAILY_DATASET, nrows=1)
        service = DailyCurtailmentService(DEFAULT_ARTIFACT)
        result = service.predict_features(frame)
        self.assertGreaterEqual(result["curtailment_event_probability"], 0)
        self.assertLessEqual(result["curtailment_event_probability"], 1)
        self.assertGreaterEqual(result["predicted_curtailment_mwh"], 0)

    def test_daily_service_rejects_tampered_model_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            altered = Path(directory) / "daily.joblib"
            shutil.copyfile(DEFAULT_ARTIFACT, altered)
            with altered.open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "checksum"):
                DailyCurtailmentService(altered).load()

    def test_dataset_coverage_separates_evaluation_dates_from_request_dates(self):
        report = json.loads(DEFAULT_REPORT.read_text(encoding="utf-8"))
        service = DailyCurtailmentService(DEFAULT_ARTIFACT, DEFAULT_REPORT)
        service.load()

        coverage = service.dataset_coverage()

        self.assertEqual(coverage["model_version"], report["model_version"])
        self.assertEqual(coverage["historical_complete_day_count"], sum(report["rows"].values()))
        self.assertEqual(coverage["historical_date_min_utc"], "2024-04-01")
        self.assertEqual(coverage["historical_date_max_utc"], "2026-08-30")
        self.assertEqual(coverage["partitions"]["test"]["complete_day_count"], report["rows"]["test"])
        self.assertEqual(coverage["requestable_date_max_utc"], datetime.now(timezone.utc).date().isoformat())
        self.assertEqual(coverage["forward_window_date_min_utc"], (datetime.now(timezone.utc).date() + timedelta(days=1)).isoformat())
        self.assertEqual(coverage["forward_window_date_max_utc"], (datetime.now(timezone.utc).date() + timedelta(days=7)).isoformat())
        self.assertEqual(coverage["maximum_forward_window_days"], 7)
        self.assertIn("not guaranteed", coverage["request_notice"].lower())


if __name__ == "__main__":
    unittest.main()
