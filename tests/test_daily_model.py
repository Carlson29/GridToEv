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
    def test_dataset_window_uses_feature_rows_without_target_labels_or_network(self):
        service = DailyCurtailmentService(DEFAULT_ARTIFACT, DEFAULT_REPORT)
        start = pd.Timestamp("2026-04-28").date()
        with patch("gridtoev.daily_model.fetch_previous_runs") as archived, \
             patch("gridtoev.daily_curtailment.fetch_live_forecast") as live, \
             patch.object(service, "predict_features", wraps=service.predict_features) as predict:
            result = service.predict_dataset_window(start, 7)
        archived.assert_not_called()
        live.assert_not_called()
        self.assertEqual(predict.call_count, 7)
        for call in predict.call_args_list:
            self.assertNotIn("curtailment_mwh", call.args[0].columns)
            self.assertNotIn("curtailment_event", call.args[0].columns)
        self.assertEqual(result["semantics"], "historical_dataset_daily_curtailment")
        self.assertEqual(result["prediction_count"], 7)
        self.assertEqual(result["predictions"][0]["target_date_utc"], start.isoformat())
        self.assertEqual(result["predictions"][-1]["target_date_utc"], "2026-05-04")
        self.assertTrue(all(
            pd.Timestamp(row["forecast_max_available_at_utc"]) <= pd.Timestamp(row["issue_timestamp_utc"])
            for row in result["predictions"]
        ))

    def test_dataset_window_rejects_out_of_coverage_and_missing_day(self):
        service = DailyCurtailmentService(DEFAULT_ARTIFACT, DEFAULT_REPORT)
        with self.assertRaisesRegex(ValueError, "dataset"):
            service.predict_dataset_window(pd.Timestamp("2026-08-30").date(), 7)
        with self.assertRaisesRegex(ValueError, "dataset"):
            service.predict_dataset_window(pd.Timestamp("2026-09-28").date(), 1)
        with self.assertRaisesRegex(ValueError, "1 and 7"):
            service.predict_dataset_window(pd.Timestamp("2026-04-28").date(), 8)
        assert service.dataset is not None
        service.dataset = service.dataset.loc[
            service.dataset["issue_timestamp_utc"].dt.date.ne(pd.Timestamp("2026-04-30").date())
        ].copy()
        with self.assertRaisesRegex(ValueError, "2026-04-30"):
            service.predict_dataset_window(pd.Timestamp("2026-04-28").date(), 7)

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

    def test_daily_service_rejects_tampered_serving_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            altered = Path(directory) / "daily.csv.gz"
            shutil.copyfile(DEFAULT_DAILY_DATASET, altered)
            with altered.open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "dataset checksum"):
                DailyCurtailmentService(DEFAULT_ARTIFACT, DEFAULT_REPORT, altered).load()

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
        self.assertEqual(coverage["window_date_min_utc"], "2024-04-01")
        self.assertEqual(coverage["window_date_max_utc"], "2026-08-30")
        self.assertEqual(coverage["maximum_window_days"], 7)
        self.assertIn("not guaranteed", coverage["request_notice"].lower())


if __name__ == "__main__":
    unittest.main()
