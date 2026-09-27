import unittest
from datetime import datetime, timedelta, timezone

import pandas as pd
from fastapi.testclient import TestClient

from gridtoev.api import create_app
from gridtoev.constants import DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH
from gridtoev.daily_curtailment import REGIONS
from gridtoev.daily_model import DailyCurtailmentService
from gridtoev.inference import PredictionService
from gridtoev.release_guard import DEFAULT_REPORT_PATH
from tests.test_raw_prediction import historical_v1_raw_request


class RawApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        v1 = PredictionService(DEFAULT_MODEL_PATH, DEFAULT_DATASET_PATH, DEFAULT_REPORT_PATH)
        cls.client = TestClient(create_app(v1, api_key="test-key", daily_service=DailyCurtailmentService()))
        cls.client.__enter__()
        cls.headers = {"X-API-Key": "test-key"}

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def test_model_catalog_and_details_report_actual_saved_estimators_and_scores(self):
        catalog = self.client.get("/models/catalog", headers=self.headers)
        self.assertEqual(catalog.status_code, 200, catalog.text)
        self.assertEqual({item["model_id"] for item in catalog.json()["models"]}, {"v1", "v2"})
        v1 = self.client.get("/model-info", headers=self.headers).json()
        self.assertEqual(v1["prediction_components"]["event_classifier"]["estimator"], "HistGradientBoostingClassifier")
        self.assertEqual(v1["prediction_components"]["dispatch_down_regressor"]["estimator"], "HistGradientBoostingRegressor")
        self.assertEqual(v1["prediction_components"]["dispatch_down_regressor"]["hyperparameters"]["loss"], "squared_error")
        self.assertEqual(v1["evaluation"]["test"]["dispatch_down_mae_mwh"], 11.825026392111367)
        self.assertEqual(v1["evaluation"]["serving_ml_weight_by_horizon"], {"30": 0.0, "60": 0.0})
        v2 = self.client.get("/model-info/daily-curtailment", headers=self.headers).json()
        self.assertEqual(v2["prediction_components"]["amount_model"]["estimator"], "ExtraTreesRegressor")
        self.assertEqual(v2["prediction_components"]["amount_model"]["hyperparameters"]["n_estimators"], 240)
        self.assertEqual(v2["evaluation"]["test"]["daily_mae_mwh"], 1723.390575132749)
        self.assertIn("not validated", v2["evaluation"]["future_window_notice"].lower())

    def test_v1_raw_schema_and_prediction_match_historical_replay(self):
        schema = self.client.get("/model-info/v1/raw-input-schema", headers=self.headers)
        self.assertEqual(schema.status_code, 200, schema.text)
        self.assertEqual(schema.json()["required_history_half_hours"], 48)
        self.assertIn("observed_dispatch_down_mwh", schema.json()["current_fields"])
        expected, current, history = historical_v1_raw_request()
        request = {
            "issue_timestamp_utc": expected["issue_timestamp_utc"],
            "forecast_horizon_minutes": 30,
            "current_observation": current,
            "history": history,
        }
        self.assertEqual(self.client.post("/predict/v1/from-raw", json=request).status_code, 401)
        result = self.client.post("/predict/v1/from-raw", json=request, headers=self.headers)
        self.assertEqual(result.status_code, 200, result.text)
        historical = self.client.post("/predict/from-dataset", json={
            "issue_timestamp_utc": expected["issue_timestamp_utc"], "forecast_horizon_minutes": 30,
        }, headers=self.headers)
        self.assertAlmostEqual(result.json()["predicted_dispatch_down_mwh"], historical.json()["predicted_dispatch_down_mwh"])
        self.assertEqual(result.json()["engineered_feature_count"], 119)
        request["current_observation"]["future_target_mwh"] = 1
        self.assertEqual(self.client.post("/predict/v1/from-raw", json=request, headers=self.headers).status_code, 422)
        del request["current_observation"]["future_target_mwh"]
        request["current_observation"]["available_at_utc"] = (
            pd.Timestamp(expected["issue_timestamp_utc"]) + pd.Timedelta(minutes=1)
        ).isoformat()
        self.assertEqual(self.client.post("/predict/v1/from-raw", json=request, headers=self.headers).status_code, 422)

    def test_v2_raw_schema_and_future_day_prediction(self):
        schema = self.client.get("/model-info/daily-curtailment/raw-input-schema", headers=self.headers)
        self.assertEqual(schema.status_code, 200, schema.text)
        self.assertEqual(schema.json()["required_hourly_rows"], 96)
        target = datetime.now(timezone.utc).date() + timedelta(days=1)
        hours = pd.date_range(target, periods=24, freq="h", tz="UTC")
        now = datetime.now(timezone.utc).isoformat()
        rows = [{
            "region": region, "target_hour_utc": hour.isoformat(),
            "forecast_available_at_utc": now, "wind_speed_100m_kmh": 20,
            "shortwave_radiation_wm2": 100, "temperature_2m_c": 12,
        } for region in REGIONS for hour in hours]
        result = self.client.post("/predict/curtailment/from-raw", json={
            "target_date_utc": target.isoformat(), "hourly_forecasts": rows,
        }, headers=self.headers)
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(result.json()["target_date_utc"], target.isoformat())
        self.assertEqual(result.json()["engineered_feature_count"], 24)
        self.assertEqual(result.json()["input_provenance"], "user_supplied_unverified")
        rows.pop()
        self.assertEqual(self.client.post("/predict/curtailment/from-raw", json={
            "target_date_utc": target.isoformat(), "hourly_forecasts": rows,
        }, headers=self.headers).status_code, 422)
        rows.append({**rows[0], "target_hour_utc": hours[-1].isoformat()})
        self.assertEqual(self.client.post("/predict/curtailment/from-raw", json={
            "target_date_utc": target.isoformat(), "hourly_forecasts": rows,
        }, headers=self.headers).status_code, 422)


if __name__ == "__main__":
    unittest.main()
