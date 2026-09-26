import unittest
from datetime import date
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from gridtoev.daily_curtailment import (
    build_daily_labels,
    build_forecast_features,
    parse_previous_runs,
    fetch_previous_runs,
)


class DailyCurtailmentDataTests(unittest.TestCase):
    def test_previous_day_forecasts_are_complete_and_available_at_issue(self):
        hours = pd.date_range("2025-06-01", periods=24, freq="h", tz="UTC")
        payload = {
            "utc_offset_seconds": 0,
            "hourly": {
                "time": [item.strftime("%Y-%m-%dT%H:%M") for item in hours],
                "wind_speed_100m_previous_day1": [20.0] * 24,
                "shortwave_radiation_previous_day1": [100.0] * 24,
                "temperature_2m_previous_day1": [12.0] * 24,
            },
        }
        parsed = parse_previous_runs(payload, "west")
        features = build_forecast_features(parsed, regions=("west",))
        self.assertEqual(len(features), 1)
        self.assertEqual(features.iloc[0]["forecast_max_available_at_utc"],
                         pd.Timestamp("2025-05-31T23:00:00Z"))
        self.assertEqual(features.iloc[0]["issue_timestamp_utc"],
                         pd.Timestamp("2025-06-01T00:00:00Z"))
        self.assertEqual(features.iloc[0]["west_wind_mean_kmh"], 20.0)

    def test_missing_hour_or_forecast_rejects_the_day(self):
        hours = pd.date_range("2025-06-01", periods=24, freq="h", tz="UTC")
        base = pd.DataFrame({
            "target_hour_utc": hours,
            "region": "west",
            "wind_speed_100m_kmh": [20.0] * 24,
            "shortwave_radiation_wm2": [100.0] * 24,
            "temperature_2m_c": [12.0] * 24,
            "available_at_utc": hours - pd.Timedelta(days=1),
        })
        self.assertTrue(build_forecast_features(base.iloc[:-1], regions=("west",)).empty)
        base.loc[0, "wind_speed_100m_kmh"] = None
        self.assertTrue(build_forecast_features(base, regions=("west",)).empty)
        base.loc[0, "wind_speed_100m_kmh"] = 20.0
        base.loc[0, "available_at_utc"] = hours[0] + pd.Timedelta(hours=1)
        self.assertTrue(build_forecast_features(base, regions=("west",)).empty)

    def test_daily_labels_require_48_valid_half_hours(self):
        times = pd.date_range("2025-06-01", periods=48, freq="30min", tz="UTC")
        raw = pd.DataFrame({
            "timestamp_utc": times,
            "curtailment_mwh": [2.0] * 48,
            "dispatch_down_available_flag": [1] * 48,
        })
        labels = build_daily_labels(raw)
        self.assertEqual(len(labels), 1)
        self.assertEqual(labels.iloc[0]["curtailment_mwh"], 96.0)
        self.assertEqual(labels.iloc[0]["curtailment_event"], 1)
        self.assertTrue(build_daily_labels(raw.iloc[:-1]).empty)
        raw.loc[2, "dispatch_down_available_flag"] = 0
        self.assertTrue(build_daily_labels(raw).empty)

    def test_download_pins_weather_model_and_separates_its_cache(self):
        payload = {
            "utc_offset_seconds": 0,
            "hourly": {
                "time": ["2025-06-01T00:00"],
                "wind_speed_100m_previous_day1": [20.0],
                "shortwave_radiation_previous_day1": [0.0],
                "temperature_2m_previous_day1": [12.0],
            },
        }
        with TemporaryDirectory() as directory, patch(
            "gridtoev.daily_curtailment._request_json", return_value=payload
        ) as request:
            fetch_previous_runs("west", date(2025, 6, 1), date(2025, 6, 1), raw_dir=Path(directory))
            self.assertEqual(request.call_args.args[0]["models"], "gfs_global")
            self.assertTrue((Path(directory) / "gfs_global" / "west" / "2025-06-01_2025-06-01.json").exists())


if __name__ == "__main__":
    unittest.main()
