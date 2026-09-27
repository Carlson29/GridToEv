"""Raw request contracts must recreate trained features without future labels."""

import unittest
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from gridtoev.constants import DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH
from gridtoev.daily_curtailment import REGIONS
from gridtoev.daily_model import DailyCurtailmentService
from gridtoev.inference import PredictionService
from gridtoev.raw_prediction import (
    V1_RAW_CURRENT_FIELDS,
    V1_RAW_HISTORY_FIELDS,
    build_v1_features_from_raw,
    build_v2_features_from_raw,
)
from gridtoev.release_guard import DEFAULT_REPORT_PATH


def historical_v1_raw_request(*, offset: int = 0, horizon: int = 30) -> tuple[pd.Series, dict, list[dict]]:
    data = pd.read_csv(DEFAULT_DATASET_PATH)
    data = data.loc[data["forecast_horizon_minutes"].eq(horizon)].reset_index(drop=True)
    rows = data.iloc[len(data) - offset - 49:len(data) - offset].reset_index(drop=True)
    current = rows.iloc[-1]
    issue = current["issue_timestamp_utc"]
    current_raw = {name: float(current[name]) for name in V1_RAW_CURRENT_FIELDS}
    current_raw.update({
        "observed_dispatch_down_mwh": float(current["dispatch_down_mwh_latest_observed"]),
        "available_at_utc": issue,
    })
    history = []
    for _, row in rows.iloc[:-1].iterrows():
        record = {name: float(row[name]) for name in V1_RAW_HISTORY_FIELDS}
        record.update({
            "timestamp_utc": row["issue_timestamp_utc"],
            "available_at_utc": row["issue_timestamp_utc"],
            "observed_dispatch_down_mwh": float(row["dispatch_down_mwh_latest_observed"]),
        })
        history.append(record)
    return current, current_raw, history


class RawPredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v1 = PredictionService(DEFAULT_MODEL_PATH, DEFAULT_DATASET_PATH, DEFAULT_REPORT_PATH)
        cls.v1.load()
        cls.v2 = DailyCurtailmentService()
        cls.v2.load()

    def test_v1_raw_feature_parity_with_frozen_dataset_row(self):
        for offset, horizon in ((0, 30), (1, 60), (25, 30), (100, 60)):
            expected, current, history = historical_v1_raw_request(offset=offset, horizon=horizon)
            features = build_v1_features_from_raw(
                expected["issue_timestamp_utc"], current, history, horizon,
            )
            columns = self.v1.bundle["feature_columns"]
            self.assertEqual(set(features), set(columns))
            for column in columns:
                self.assertTrue(np.isclose(features[column], expected[column], rtol=1e-8, atol=1e-8), column)
            raw_prediction = self.v1.predict_features(
                {key: value for key, value in features.items() if key != "forecast_horizon_minutes"},
                horizon, issue_timestamp_utc=expected["issue_timestamp_utc"],
            )
            dataset_prediction = self.v1.predict_from_dataset(expected["issue_timestamp_utc"], horizon)
            self.assertAlmostEqual(raw_prediction["predicted_dispatch_down_mwh"], dataset_prediction["predicted_dispatch_down_mwh"])

    def test_v1_raw_rejects_gaps_and_values_unavailable_at_issue(self):
        expected, current, history = historical_v1_raw_request()
        with self.assertRaisesRegex(ValueError, "48 consecutive"):
            build_v1_features_from_raw(expected["issue_timestamp_utc"], current, history[:-1], 30)
        current["available_at_utc"] = (pd.Timestamp(expected["issue_timestamp_utc"]) + pd.Timedelta(minutes=1)).isoformat()
        with self.assertRaisesRegex(ValueError, "available"):
            build_v1_features_from_raw(expected["issue_timestamp_utc"], current, history, 30)
        current["available_at_utc"] = expected["issue_timestamp_utc"]
        if pd.Timestamp(expected["issue_timestamp_utc"]).minute == 30:
            current["entsoe_price_eur_mwh"] += 1
            with self.assertRaisesRegex(ValueError, "preceding"):
                build_v1_features_from_raw(expected["issue_timestamp_utc"], current, history, 30)

    def test_v2_raw_hourly_forecasts_build_same_daily_features_and_predict(self):
        target_date = (datetime.now(timezone.utc) + timedelta(days=1)).date()
        issued_at = datetime.now(timezone.utc)
        hours = pd.date_range(target_date, periods=24, freq="h", tz="UTC")
        records = [
            {
                "region": region,
                "target_hour_utc": hour.isoformat(),
                "forecast_available_at_utc": issued_at.isoformat(),
                "wind_speed_100m_kmh": 20.0 + index,
                "shortwave_radiation_wm2": 100.0,
                "temperature_2m_c": 12.0,
            }
            for region in REGIONS for index, hour in enumerate(hours)
        ]
        features = build_v2_features_from_raw(target_date, records, issued_at)
        self.assertEqual(len(features), 1)
        self.assertEqual(set(self.v2.bundle["metadata"]["feature_columns"]) - set(features), set())
        prediction = self.v2.predict_features(features, issued_at_utc=issued_at)
        self.assertEqual(prediction["target_date_utc"], target_date.isoformat())
        self.assertGreaterEqual(prediction["predicted_curtailment_mwh"], 0)

        with self.assertRaisesRegex(ValueError, "complete"):
            build_v2_features_from_raw(target_date, records[:-1], issued_at)
        records[0]["forecast_available_at_utc"] = (issued_at + timedelta(minutes=1)).isoformat()
        with self.assertRaisesRegex(ValueError, "available"):
            build_v2_features_from_raw(target_date, records, issued_at)


if __name__ == "__main__":
    unittest.main()
