import sys
import tempfile
import unittest
import warnings
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Using `httpx` with `starlette.testclient` is deprecated.*")
    from fastapi.testclient import TestClient

from gridtoev.api import create_app
from gridtoev.inference import PredictionService
from gridtoev.training import TrainingConfig, train_and_save
from tests.test_training_and_inference import make_synthetic_dataset


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        temp_path = Path(cls.temp_dir.name)
        cls.dataset_path = temp_path / "dataset.csv"
        cls.artifact_path = temp_path / "model.joblib"
        cls.data = make_synthetic_dataset(cls.dataset_path)
        cls.training_result = train_and_save(
            dataset_path=cls.dataset_path,
            artifact_path=cls.artifact_path,
            config=TrainingConfig(max_iter=10, min_samples_leaf=5, random_state=9),
        )
        service = PredictionService(cls.artifact_path, cls.dataset_path)
        cls.client = TestClient(create_app(service))
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.__exit__(None, None, None)
        cls.temp_dir.cleanup()

    def test_health_and_model_info(self) -> None:
        health = self.client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "ready")

        model_info = self.client.get("/model-info")
        self.assertEqual(model_info.status_code, 200)
        self.assertEqual(model_info.json()["forecast_horizons_minutes"], [30, 60])

    def test_predict_from_dataset(self) -> None:
        timestamp = self.data["issue_timestamp_utc"].iloc[-30].isoformat()
        response = self.client.post(
            "/predict/from-dataset",
            json={
                "issue_timestamp_utc": timestamp,
                "forecast_horizon_minutes": 30,
                "flexible_load_capacity_mw": 100.0,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertIn("dispatch_down_probability", body)
        self.assertIn("predicted_dispatch_down_mwh", body)

    def test_latest_returns_both_horizons(self) -> None:
        response = self.client.get("/predict/latest")
        self.assertEqual(response.status_code, 200, response.text)
        predictions = response.json()["predictions"]
        self.assertEqual({item["forecast_horizon_minutes"] for item in predictions}, {30, 60})

    def test_predict_from_complete_feature_snapshot(self) -> None:
        row = self.data.iloc[-30]
        features = {
            column: float(row[column])
            for column in self.training_result.bundle["feature_columns"]
            if column != "forecast_horizon_minutes"
        }
        response = self.client.post(
            "/predict/features",
            json={
                "issue_timestamp_utc": row["issue_timestamp_utc"].isoformat(),
                "forecast_horizon_minutes": 30,
                "flexible_load_capacity_mw": 100.0,
                "features": features,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("prediction_interval_p90_mwh", response.json())

    def test_unknown_dataset_timestamp_returns_not_found(self) -> None:
        response = self.client.post(
            "/predict/from-dataset",
            json={
                "issue_timestamp_utc": "2030-01-01T00:00:00Z",
                "forecast_horizon_minutes": 30,
                "flexible_load_capacity_mw": 100.0,
            },
        )
        self.assertEqual(response.status_code, 404)

    def test_optional_daily_model_does_not_change_v1_when_unconfigured(self) -> None:
        response = self.client.post(
            "/predict/curtailment/day", json={"target_date_utc": "2026-09-26"}
        )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(self.client.get("/health").status_code, 200)

    def test_daily_endpoint_has_a_separate_prediction_contract(self) -> None:
        class FakeDailyService:
            bundle = {"metadata": {"model_version": "2.0.0-daily-experimental"}}

            def load(self):
                pass

            def predict_date(self, target_date):
                return {
                    "model_version": "2.0.0-daily-experimental",
                    "experimental": True,
                    "target_date_utc": target_date.isoformat(),
                    "issue_timestamp_utc": f"{target_date.isoformat()}T00:00:00+00:00",
                    "forecast_max_available_at_utc": "2026-09-25T23:00:00+00:00",
                    "curtailment_event_probability": 0.7,
                    "predicted_curtailment_mwh": 1200.0,
                    "target": "Total EirGrid curtailment during this UTC day",
                }

        with TestClient(create_app(self.client.app.state.prediction_service, FakeDailyService())) as client:
            response = client.post(
                "/predict/curtailment/day", json={"target_date_utc": "2026-09-26"}
            )
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["target_date_utc"], "2026-09-26")
            self.assertEqual(client.get("/model-info/daily-curtailment").json()["model_version"],
                             "2.0.0-daily-experimental")


if __name__ == "__main__":
    unittest.main()
