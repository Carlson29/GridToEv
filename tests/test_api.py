import sys
import tempfile
import unittest
import warnings
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Using `httpx` with `starlette.testclient` is deprecated.*")
    from fastapi.testclient import TestClient

from gridtoev.api import create_app
from gridtoev.actuals import ActualsService
from gridtoev.daily_model import DailyCurtailmentService
from gridtoev.inference import FeatureValidationError, PredictionService
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
        self.assertFalse(health.json()["daily_model_available"])

        root = self.client.get("/")
        self.assertEqual(root.status_code, 200)
        self.assertEqual(root.json()["docs_url"], "/docs")

        model_info = self.client.get("/model-info")
        self.assertEqual(model_info.status_code, 200)
        self.assertEqual(model_info.json()["forecast_horizons_minutes"], [30, 60])
        self.assertEqual(model_info.json()["model_artifact"], "model.joblib")
        self.assertNotIn("model_path", model_info.json())

        dataset_info = self.client.get("/dataset/info")
        self.assertEqual(dataset_info.status_code, 200, dataset_info.text)
        info = dataset_info.json()
        self.assertEqual(info["timezone"], "UTC")
        self.assertEqual(info["interval_minutes"], 30)
        self.assertEqual(info["supported_forecast_horizons_minutes"], [30, 60])
        self.assertIn("T", info["timestamp_example"])
        self.assertTrue(info["timestamp_example"].endswith("Z"))
        self.assertLess(
            info["available_issue_timestamp_min_utc"],
            info["available_issue_timestamp_max_utc"],
        )

    def test_openapi_explains_model_routes_and_inputs(self) -> None:
        schema = self.client.get("/openapi.json").json()
        self.assertIn("30/60-minute", schema["info"]["description"])
        self.assertIn("daily curtailment", schema["info"]["description"].lower())
        expected_tags = {
            "/health": "Service",
            "/model-info": "V1 — 30/60-minute model",
            "/model-info/v1/formulas": "V1 — 30/60-minute model",
            "/model-info/v1/fitted-formulas": "V1 — 30/60-minute model",
            "/model-info/v1/fitted-formulas/{estimator_id}/trees/{tree_index}": "V1 — 30/60-minute model",
            "/dataset/info": "V1 — 30/60-minute model",
            "/dataset/available-times": "V1 — 30/60-minute model",
            "/predict/latest": "V1 — 30/60-minute model",
            "/predict/from-dataset": "V1 — 30/60-minute model",
            "/predict/window/from-dataset": "V1 — 30/60-minute model",
            "/predict/features": "V1 — 30/60-minute model",
            "/model-info/daily-curtailment": "V2 — daily curtailment model",
            "/model-info/daily-curtailment/formulas": "V2 — daily curtailment model",
            "/model-info/daily-curtailment/fitted-formulas": "V2 — daily curtailment model",
            "/model-info/daily-curtailment/fitted-formulas/{estimator_id}/trees/{tree_index}": "V2 — daily curtailment model",
            "/dataset/daily-curtailment/coverage": "V2 — daily curtailment model",
            "/predict/curtailment/day": "V2 — daily curtailment model",
            "/predict/curtailment/window": "V2 — daily curtailment model",
            "/predict/curtailment/from-raw": "V2 — daily curtailment model",
            "/predict/v1/from-raw": "V1 — 30/60-minute model",
            "/model-info/v1/raw-input-schema": "V1 — 30/60-minute model",
            "/model-info/daily-curtailment/raw-input-schema": "V2 — daily curtailment model",
            "/models/catalog": "Service",
            "/models/about": "Service",
            "/actuals/coverage": "Observed outcomes",
            "/actuals/v1": "Observed outcomes",
            "/actuals/v1/batch": "Observed outcomes",
            "/actuals/v1/window": "Observed outcomes",
            "/actuals/daily-curtailment": "Observed outcomes",
            "/actuals/daily-curtailment/window": "Observed outcomes",
            "/actuals/curtailment/sources": "Observed curtailment by source (Wind/Solar)",
            "/actuals/curtailment/sources/coverage": "Observed curtailment by source (Wind/Solar)",
            "/predict/curtailment/sources/day": "V2 — wind/solar split (experimental)",
            "/model-info/curtailment/sources": "V2 — wind/solar split (experimental)",
        }
        self.assertEqual(set(schema["paths"]), set(expected_tags))
        for path, expected_tag in expected_tags.items():
            operation = next(iter(schema["paths"][path].values()))
            self.assertEqual(operation["tags"], [expected_tag], path)
            self.assertTrue(operation["summary"], path)
            self.assertTrue(operation["description"], path)
        self.assertIn("historical", schema["paths"]["/predict/latest"]["get"]["description"].lower())
        self.assertIn("not a day-ahead", schema["paths"]["/predict/window/from-dataset"]["post"]["description"].lower())
        self.assertIn("YYYY-MM-DD", schema["components"]["schemas"]["DailyCurtailmentRequest"]["properties"]["target_date_utc"]["description"])
        self.assertEqual(schema["components"]["schemas"]["DatasetWindowPredictionRequest"]["properties"]["duration_hours"]["maximum"], 24)

    def test_daily_dataset_coverage_requires_key_and_shows_historical_and_request_ranges(self) -> None:
        service = PredictionService(self.artifact_path, self.dataset_path)
        with TestClient(create_app(service, api_key="team-secret", daily_service=DailyCurtailmentService())) as client:
            self.assertEqual(client.get("/dataset/daily-curtailment/coverage").status_code, 401)
            response = client.get(
                "/dataset/daily-curtailment/coverage", headers={"X-API-Key": "team-secret"}
            )
            self.assertEqual(response.status_code, 200, response.text)
            result = response.json()
            self.assertEqual(result["historical_complete_day_count"], 882)
            self.assertEqual(result["partitions"]["train"]["complete_day_count"], 275)
            self.assertEqual(result["requestable_date_max_utc"], datetime.now(timezone.utc).date().isoformat())

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

    def test_feature_snapshot_rejects_extra_and_nonfinite_fields(self) -> None:
        row = self.data.iloc[-30]
        features = {
            column: float(row[column])
            for column in self.training_result.bundle["feature_columns"]
            if column != "forecast_horizon_minutes"
        }
        request = {
            "issue_timestamp_utc": row["issue_timestamp_utc"].isoformat(),
            "forecast_horizon_minutes": 30,
            "features": {**features, "unknown_signal": 1.0},
        }
        self.assertEqual(self.client.post("/predict/features", json=request).status_code, 422)
        service = PredictionService(self.artifact_path, self.dataset_path)
        service.load()
        with self.assertRaisesRegex(FeatureValidationError, "Non-finite"):
            service.predict_features(
                {**features, "dispatch_down_mwh_lag_1": math.inf}, 30
            )

    def test_model_info_exposes_feature_contract_not_server_path(self) -> None:
        response = self.client.get("/model-info")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["feature_contract"]["feature_count"], len(self.training_result.bundle["feature_columns"]))
        self.assertNotIn("model_path", response.json())

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
        detail = response.json()["detail"]
        self.assertEqual(detail["error"], "dataset_timestamp_not_available")
        self.assertIn("expected_format", detail)
        self.assertIn("available_issue_timestamp_min_utc", detail)
        self.assertIn("nearest_before_utc", detail)

    def test_dataset_timestamp_requires_timezone(self) -> None:
        response = self.client.post(
            "/predict/from-dataset",
            json={
                "issue_timestamp_utc": "2026-01-01T10:30:00",
                "forecast_horizon_minutes": 30,
            },
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("timezone", response.text.lower())

    def test_openapi_documents_date_guidance_and_window_endpoint(self) -> None:
        schema = self.client.get("/openapi.json").json()
        self.assertIn("/dataset/info", schema["paths"])
        self.assertIn("/predict/window/from-dataset", schema["paths"])
        request_schema = schema["components"]["schemas"]["DatasetPredictionRequest"]
        timestamp_schema = request_schema["properties"]["issue_timestamp_utc"]
        self.assertEqual(timestamp_schema["format"], "date-time")
        self.assertIn("/dataset/info", timestamp_schema["description"])
        self.assertEqual(timestamp_schema["examples"], ["2026-01-15T12:30:00Z"])

    def test_two_hour_window_returns_ordered_array_and_horizon_summaries(self) -> None:
        issue_times = self.data["issue_timestamp_utc"].drop_duplicates().sort_values()
        issue_times = issue_times.reset_index(drop=True)
        start = issue_times.iloc[20]
        response = self.client.post(
            "/predict/window/from-dataset",
            json={
                "start_timestamp_utc": start.isoformat(),
                "duration_hours": 2,
                "forecast_horizons_minutes": [30, 60],
                "flexible_load_capacity_mw": 100,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["semantics"], "historical_rolling_short_horizon")
        self.assertEqual(body["prediction_count"], 8)
        self.assertEqual(len(body["predictions"]), 8)
        self.assertEqual(set(body["summary_by_horizon"]), {"30", "60"})
        self.assertEqual(body["summary_by_horizon"]["30"]["interval_count"], 4)
        order = [
            (item["issue_timestamp_utc"], item["forecast_horizon_minutes"])
            for item in body["predictions"]
        ]
        self.assertEqual(order, sorted(order))

    def test_one_day_window_and_out_of_range_guidance(self) -> None:
        issue_times = self.data["issue_timestamp_utc"].drop_duplicates().sort_values()
        issue_times = issue_times.reset_index(drop=True)
        valid = self.client.post(
            "/predict/window/from-dataset",
            json={
                "start_timestamp_utc": issue_times.iloc[5].isoformat(),
                "duration_hours": 24,
                "forecast_horizons_minutes": [30],
            },
        )
        self.assertEqual(valid.status_code, 200, valid.text)
        self.assertEqual(valid.json()["prediction_count"], 48)
        self.assertEqual(valid.json()["duration_hours"], 24)
        self.assertEqual(self.client.get("/dataset/info").json()["maximum_window_hours"], 24)
        too_long = self.client.post(
            "/predict/window/from-dataset",
            json={"start_timestamp_utc": issue_times.iloc[5].isoformat(), "duration_hours": 24.5},
        )
        self.assertEqual(too_long.status_code, 422)
        with self.assertRaisesRegex(FeatureValidationError, "between 0.5 and 24"):
            self.client.app.state.prediction_service.predict_window_from_dataset(
                issue_times.iloc[5], 24.5, [30],
            )

        invalid = self.client.post(
            "/predict/window/from-dataset",
            json={
                "start_timestamp_utc": issue_times.iloc[-2].isoformat(),
                "duration_hours": 2,
                "forecast_horizons_minutes": [30, 60],
            },
        )
        self.assertEqual(invalid.status_code, 404)
        detail = invalid.json()["detail"]
        self.assertEqual(detail["error"], "dataset_window_not_available")
        self.assertIn("latest_valid_start_utc_for_duration", detail)

        example = self.client.get("/openapi.json").json()["paths"]["/predict/window/from-dataset"]["post"]["requestBody"]["content"]["application/json"]["example"]
        self.assertEqual(example["duration_hours"], 24)
        example_response = self.client.post("/predict/window/from-dataset", json=example)
        self.assertEqual(example_response.status_code, 200, example_response.text)
        self.assertEqual(example_response.json()["semantics"], "historical_rolling_short_horizon")

    def test_optional_api_key_protects_data_and_prediction_routes(self) -> None:
        service = PredictionService(self.artifact_path, self.dataset_path)
        with TestClient(create_app(service, api_key="team-secret")) as secured_client:
            self.assertEqual(secured_client.get("/health").status_code, 200)
            self.assertEqual(secured_client.get("/").status_code, 200)

            missing = secured_client.get("/predict/latest")
            self.assertEqual(missing.status_code, 401)
            self.assertEqual(missing.headers["www-authenticate"], "ApiKey")

            wrong = secured_client.get(
                "/predict/latest",
                headers={"X-API-Key": "wrong-secret"},
            )
            self.assertEqual(wrong.status_code, 401)

            accepted = secured_client.get(
                "/predict/latest",
                headers={"X-API-Key": "team-secret"},
            )
            self.assertEqual(accepted.status_code, 200, accepted.text)

            model_info = secured_client.get(
                "/model-info",
                headers={"X-API-Key": "team-secret"},
            )
            self.assertEqual(model_info.status_code, 200, model_info.text)

    def test_existing_positional_api_key_argument_stays_protected(self) -> None:
        service = PredictionService(self.artifact_path, self.dataset_path)
        with TestClient(create_app(service, "team-secret")) as client:
            self.assertEqual(client.get("/predict/latest").status_code, 401)
            self.assertEqual(client.get("/predict/latest", headers={"X-API-Key": "team-secret"}).status_code, 200)

    def test_actual_lookup_routes_are_keyed_to_targets_and_protected(self) -> None:
        class FakeActuals:
            def coverage(self):
                return {"source": "test", "available_target_timestamp_min_utc": "2026-01-01T00:00:00+00:00",
                        "available_target_timestamp_max_utc": "2026-01-01T23:30:00+00:00",
                        "complete_day_min_utc": "2026-01-01", "complete_day_max_utc": "2026-01-01",
                        "notice": "snapshot"}

            def point(self, target):
                return {"status": "available", "target_timestamp_utc": target.isoformat(),
                        "source_latest_timestamp_utc": "2026-01-01T23:30:00+00:00",
                        "actual_dispatch_down_mwh": 3.0, "actual_curtailment_mwh": 2.0,
                        "actual_constraint_mwh": 1.0, "actual_dispatch_down_event": True}

            def points(self, targets):
                return [self.point(target) for target in targets]

            def day(self, target):
                return {"status": "available", "target_date_utc": target.isoformat(),
                        "source_latest_timestamp_utc": "2026-01-01T23:30:00+00:00",
                        "actual_curtailment_mwh": 96.0, "actual_curtailment_event": True,
                        "complete_half_hour_count": 48}

        service = PredictionService(self.artifact_path, self.dataset_path)
        with TestClient(create_app(service, api_key="team-secret", actuals_service=FakeActuals())) as client:
            self.assertEqual(client.get("/actuals/v1", params={"target_timestamp_utc": "2026-01-01T00:30:00Z"}).status_code, 401)
            headers = {"X-API-Key": "team-secret"}
            actual = client.get("/actuals/v1", params={"target_timestamp_utc": "2026-01-01T00:30:00Z"}, headers=headers)
            self.assertEqual(actual.status_code, 200, actual.text)
            self.assertEqual(actual.json()["actual_dispatch_down_mwh"], 3.0)
            self.assertEqual(client.get("/actuals/daily-curtailment", params={"target_date_utc": "2026-01-01"}, headers=headers).json()["actual_curtailment_mwh"], 96.0)
            batch = client.post("/actuals/v1/batch", json={"target_timestamps_utc": ["2026-01-01T00:30:00Z", "2026-01-01T01:00:00Z"]}, headers=headers)
            self.assertEqual(batch.status_code, 200, batch.text)
            self.assertEqual(batch.json()["count"], 2)
            coverage_response = client.get("/actuals/coverage", headers=headers)
            self.assertEqual(coverage_response.status_code, 200)
            coverage = coverage_response.json()
            self.assertEqual(coverage["coverage_kind"], "observed_outcomes_archive")
            self.assertEqual(coverage["available_target_timestamp_min_utc"], "2026-01-01T00:00:00+00:00")
            v1_dataset = client.get("/dataset/info", headers=headers).json()
            self.assertEqual(
                coverage["v1_prediction_dataset"]["available_issue_timestamp_min_utc"],
                v1_dataset["available_issue_timestamp_min_utc"],
            )
            self.assertEqual(
                coverage["v1_prediction_dataset"]["available_issue_timestamp_max_utc"],
                v1_dataset["available_issue_timestamp_max_utc"],
            )
            self.assertEqual(coverage["v1_prediction_dataset"]["coverage_endpoint"], "/dataset/info")
            self.assertIn("not V1 prediction-input coverage", coverage["coverage_notice"])
            self.assertEqual(client.get("/actuals/v1", params={"target_timestamp_utc": "2026-01-01T00:30:00"}, headers=headers).status_code, 422)

    def test_missing_actuals_archive_does_not_affect_v1(self) -> None:
        service = PredictionService(self.artifact_path, self.dataset_path)
        missing = self.dataset_path.parent / "no_actuals_here.csv.gz"
        with TestClient(create_app(service, actuals_service=ActualsService(missing))) as client:
            self.assertEqual(client.get("/health").status_code, 200)
            response = client.get("/actuals/coverage")
            self.assertEqual(response.status_code, 503)
            self.assertNotIn(str(missing), response.text)
            v1_window = client.post("/actuals/v1/window", json={
                "start_target_timestamp_utc": "2026-01-01T00:00:00Z", "duration_hours": 1,
            })
            self.assertEqual(v1_window.status_code, 503)
            self.assertNotIn(str(missing), v1_window.text)
            v2_window = client.post("/actuals/daily-curtailment/window", json={
                "start_date_utc": "2026-01-01", "days": 2,
            })
            self.assertEqual(v2_window.status_code, 503)
            self.assertNotIn(str(missing), v2_window.text)

    def test_actual_windows_return_ordered_outcomes_without_model_predictions(self) -> None:
        class WindowActuals:
            def point(self, target):
                status = "available" if target.hour == 0 else "pending"
                return {
                    "status": status, "target_timestamp_utc": target.isoformat(),
                    "source_latest_timestamp_utc": "2026-01-01T00:30:00+00:00",
                    "actual_dispatch_down_mwh": 3.0 if status == "available" else None,
                    "actual_curtailment_mwh": 2.0 if status == "available" else None,
                    "actual_constraint_mwh": 1.0 if status == "available" else None,
                    "actual_dispatch_down_event": True if status == "available" else None,
                }

            def points(self, targets):
                return [self.point(target) for target in targets]

            def day(self, target):
                status = "available" if target.isoformat() == "2026-01-01" else "pending"
                return {
                    "status": status, "target_date_utc": target.isoformat(),
                    "source_latest_timestamp_utc": "2026-01-01T23:30:00+00:00",
                    "actual_curtailment_mwh": 96.0 if status == "available" else None,
                    "actual_curtailment_event": True if status == "available" else None,
                    "complete_half_hour_count": 48 if status == "available" else 0,
                }

        service = PredictionService(self.artifact_path, self.dataset_path)
        with TestClient(create_app(service, api_key="team-secret", actuals_service=WindowActuals())) as client:
            v1_path = "/actuals/v1/window"
            v1_body = {"start_target_timestamp_utc": "2026-01-01T00:00:00Z", "duration_hours": 1.5}
            self.assertEqual(client.post(v1_path, json=v1_body).status_code, 401)
            headers = {"X-API-Key": "team-secret"}
            v1 = client.post(v1_path, json=v1_body, headers=headers)
            self.assertEqual(v1.status_code, 200, v1.text)
            self.assertEqual(v1.json()["actual_count"], 3)
            self.assertEqual(v1.json()["status_counts"], {"available": 2, "pending": 1, "missing": 0})
            self.assertEqual(
                [row["target_timestamp_utc"] for row in v1.json()["actuals"]],
                ["2026-01-01T00:00:00+00:00", "2026-01-01T00:30:00+00:00", "2026-01-01T01:00:00+00:00"],
            )
            self.assertIsNone(v1.json()["actuals"][-1]["actual_dispatch_down_mwh"])
            full_span = client.post(v1_path, json={**v1_body, "duration_hours": 24.5}, headers=headers)
            self.assertEqual(full_span.status_code, 200, full_span.text)
            self.assertEqual(full_span.json()["actual_count"], 49)
            self.assertEqual(client.post(v1_path, json={**v1_body, "duration_hours": 25}, headers=headers).status_code, 422)
            self.assertEqual(client.post(v1_path, json={**v1_body, "start_target_timestamp_utc": "2026-01-01T00:15:00Z"}, headers=headers).status_code, 422)
            self.assertEqual(client.post(v1_path, json={**v1_body, "start_target_timestamp_utc": "9999-12-31T23:30:00Z"}, headers=headers).status_code, 422)

            v2_path = "/actuals/daily-curtailment/window"
            v2_body = {"start_date_utc": "2026-01-01", "days": 2}
            self.assertEqual(client.post(v2_path, json=v2_body).status_code, 401)
            v2 = client.post(v2_path, json=v2_body, headers=headers)
            self.assertEqual(v2.status_code, 200, v2.text)
            self.assertEqual(v2.json()["actual_count"], 2)
            self.assertEqual(v2.json()["status_counts"], {"available": 1, "pending": 1, "missing": 0})
            self.assertEqual([row["target_date_utc"] for row in v2.json()["actuals"]], ["2026-01-01", "2026-01-02"])
            self.assertIsNone(v2.json()["actuals"][-1]["actual_curtailment_mwh"])
            self.assertEqual(client.post(v2_path, json={**v2_body, "days": 8}, headers=headers).status_code, 422)
            self.assertEqual(client.post(v2_path, json={**v2_body, "start_date_utc": "9999-12-31"}, headers=headers).status_code, 422)

            schema = client.get("/openapi.json").json()["paths"]
            self.assertIn(v1_path, schema)
            self.assertIn(v2_path, schema)

    def test_optional_daily_model_does_not_change_v1_when_unconfigured(self) -> None:
        about = self.client.get("/models/about")
        self.assertEqual(about.status_code, 200, about.text)
        models = {item["model_id"]: item for item in about.json()["models"]}
        self.assertTrue(models["v1"]["available"])
        self.assertFalse(models["v2"]["available"])
        self.assertIsNone(models["v2"]["model_version"])
        self.assertIsNone(models["v2"]["evaluation"])
        response = self.client.post(
            "/predict/curtailment/day", json={"target_date_utc": "2026-09-26"}
        )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(self.client.get("/dataset/daily-curtailment/coverage").status_code, 503)
        self.assertEqual(self.client.get("/model-info/daily-curtailment/raw-input-schema").status_code, 503)
        self.assertEqual(self.client.get("/model-info/daily-curtailment/formulas").status_code, 503)
        self.assertEqual(self.client.get("/model-info/daily-curtailment/fitted-formulas").status_code, 503)
        self.assertEqual(self.client.get("/model-info/daily-curtailment/fitted-formulas/event_model/trees/0").status_code, 503)
        self.assertFalse(self.client.get("/models/catalog").json()["models"][1]["available"])
        self.assertEqual(self.client.get("/health").status_code, 200)

    def test_broken_optional_daily_model_cannot_take_v1_offline(self) -> None:
        class BrokenDailyService:
            def load(self):
                raise ValueError("bad optional artifact")

        service = PredictionService(self.artifact_path, self.dataset_path)
        with TestClient(create_app(service, daily_service=BrokenDailyService())) as client:
            self.assertEqual(client.get("/health").status_code, 200)
            self.assertFalse(client.get("/health").json()["daily_model_available"])
            self.assertEqual(client.get("/predict/latest").status_code, 200)
            self.assertEqual(client.post("/predict/curtailment/day", json={"target_date_utc": "2026-01-15"}).status_code, 503)

    def test_daily_endpoint_has_a_separate_prediction_contract(self) -> None:
        class FakeDailyService:
            bundle = {"metadata": {"model_version": "2.0.0-daily-experimental"}}

            def load(self):
                pass

            def model_info(self):
                return self.bundle["metadata"]

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

        with TestClient(create_app(self.client.app.state.prediction_service, daily_service=FakeDailyService())) as client:
            self.assertTrue(client.get("/health").json()["daily_model_available"])
            response = client.post(
                "/predict/curtailment/day", json={"target_date_utc": "2026-09-26"}
            )
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["target_date_utc"], "2026-09-26")
            self.assertEqual(client.get("/model-info/daily-curtailment").json()["model_version"],
                             "2.0.0-daily-experimental")

    def test_daily_dataset_window_accepts_april_example_and_rejects_future(self) -> None:
        service = PredictionService(self.artifact_path, self.dataset_path)
        with TestClient(create_app(service, api_key="secret", daily_service=DailyCurtailmentService())) as client:
            payload = {"start_date_utc": "2026-04-28", "days": 7}
            self.assertEqual(client.post("/predict/curtailment/window", json=payload).status_code, 401)
            response = client.post("/predict/curtailment/window", json=payload, headers={"X-API-Key": "secret"})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["semantics"], "historical_dataset_daily_curtailment")
            self.assertEqual(response.json()["start_date_utc"], "2026-04-28")
            self.assertEqual(response.json()["end_date_utc"], "2026-05-04")
            self.assertEqual(response.json()["prediction_count"], 7)
            self.assertEqual(client.post("/predict/curtailment/window", json={**payload, "days": 8}, headers={"X-API-Key": "secret"}).status_code, 422)
            future = client.post(
                "/predict/curtailment/window",
                json={"start_date_utc": "2026-09-28", "days": 7},
                headers={"X-API-Key": "secret"},
            )
            self.assertEqual(future.status_code, 422, future.text)
            self.assertIn("dataset", future.text.lower())

            example = client.get("/openapi.json").json()["paths"]["/predict/curtailment/window"]["post"]["requestBody"]["content"]["application/json"]["example"]
            self.assertEqual(example, payload)


if __name__ == "__main__":
    unittest.main()
