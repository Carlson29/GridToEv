import tempfile
import unittest
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gridtoev.daily_model import (
    DEFAULT_ARTIFACT, DEFAULT_REPORT, DailyCurtailmentService,
    _validate_dataset, chronological_partitions, feature_columns,
)
from gridtoev.daily_curtailment import DEFAULT_DAILY_DATASET


class DailyModelTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
