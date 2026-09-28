import json
import shutil
import tempfile
import unittest
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Using `httpx` with `starlette.testclient` is deprecated.*")
    from fastapi.testclient import TestClient

from gridtoev import daily_model
from gridtoev.api import create_app
from gridtoev.daily_model import DailyCurtailmentService
from gridtoev.source_allocation import (
    PHYSICS_ARTIFACT,
    PHYSICS_REPORT,
    add_physics_features,
    fit_physics_share,
    load_dataset,
)
from gridtoev.source_prediction import CapacityUnavailableError, SourceAllocationService


ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / "data/processed/eirgrid_core_history_30min.csv.gz"
FEATURE_COLUMNS = ["issue_timestamp_utc", "forecast_max_available_at_utc"]


def _dataset_rows(first: str, count: int) -> pd.DataFrame:
    data = daily_model._validate_dataset(pd.read_csv(daily_model.DEFAULT_DAILY_DATASET))
    rows = data.loc[data["issue_timestamp_utc"].ge(first)].head(count)
    # Only forecast/calendar inputs, exactly as the live route builds them; never labels.
    return rows[FEATURE_COLUMNS + daily_model.feature_columns(rows)].reset_index(drop=True)


class DatasetBackedSplit(SourceAllocationService):
    """Serve from bundled archived-forecast rows instead of the network."""

    def predict_date(self, target_date: date) -> dict:
        self.check_capacity(target_date)
        features = _dataset_rows(target_date.isoformat(), 1)
        return self.split(features, self.daily_service.predict_features(features))


class SourcePredictionServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.daily = DailyCurtailmentService()
        cls.daily.load()
        cls.service = SourceAllocationService(cls.daily)
        cls.service.load()

    def test_served_split_matches_the_evaluated_model_exactly(self) -> None:
        data = load_dataset()
        history = pd.read_csv(HISTORY, usecols=["timestamp_utc", "eirgrid_ie_solar_availability_mw", "eirgrid_ie_wind_availability_mw"])
        development = add_physics_features(data.loc[data["issue_timestamp_utc"].lt("2026-01-01T00:00:00Z")], history)
        evaluated = fit_physics_share(development)
        rows = _dataset_rows("2026-03-01", 5)
        expected = evaluated.predict_proba(add_physics_features(rows, history)[["physics_log_ratio"]])[:, 1]
        for index in range(len(rows)):
            features = rows.iloc[[index]].reset_index(drop=True)
            parent = self.daily.predict_features(features)
            served = self.service.split(features, parent)
            self.assertAlmostEqual(served["predicted_wind_share_percent"], round(100 * expected[index], 2), places=6)
            self.assertAlmostEqual(
                served["predicted_wind_curtailment_mwh"] + served["predicted_solar_curtailment_mwh"],
                parent["predicted_curtailment_mwh"], places=9,
            )
            self.assertGreaterEqual(served["predicted_solar_curtailment_mwh"], 0)

    def test_inputs_are_available_before_issue(self) -> None:
        features = _dataset_rows("2026-05-10", 1)
        result = self.service.split(features, self.daily.predict_features(features))
        issue = pd.Timestamp(result["issue_timestamp_utc"])
        self.assertLessEqual(pd.Timestamp(result["forecast_max_available_at_utc"]), issue)
        # Capacity data must end before the previous month starts (month M-2 rule).
        through = pd.Timestamp(result["capacity_proxy"]["published_data_through_utc"], tz="UTC")
        self.assertLess(through, (issue.tz_localize(None).to_period("M") - 1).start_time.tz_localize("UTC"))
        self.assertTrue(result["experimental"])

    def test_dates_beyond_published_capacity_are_refused_not_guessed(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            short = Path(folder) / "history.csv"
            history = pd.read_csv(HISTORY, usecols=["timestamp_utc", "eirgrid_ie_solar_availability_mw", "eirgrid_ie_wind_availability_mw"])
            history.loc[history["timestamp_utc"].lt("2026-07-01")].to_csv(short, index=False)
            service = SourceAllocationService(self.daily, history_path=short)
            service.load()
            self.assertEqual(service.latest_predictable_date(), date(2026, 8, 31))
            with self.assertRaisesRegex(CapacityUnavailableError, "archive must be refreshed"):
                service.check_capacity(date(2026, 9, 1))

    def test_tampered_artifact_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            artifact = Path(folder) / "artifact.json"
            shutil.copy(PHYSICS_ARTIFACT, artifact)
            artifact.write_text(artifact.read_text(encoding="utf-8").replace('"slope": 0.', '"slope": 9.'), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum"):
                SourceAllocationService(self.daily, artifact_path=artifact).load()

    def test_artifact_is_frozen_before_the_prospective_period(self) -> None:
        artifact = json.loads(PHYSICS_ARTIFACT.read_text(encoding="utf-8"))
        report = json.loads(PHYSICS_REPORT.read_text(encoding="utf-8"))
        self.assertEqual(artifact["fitted_on"], ["2024-04-01", "2025-12-31"])
        self.assertLess(artifact["fitted_on"][1], report["release_gate"]["rule"].split("fresh days from ")[1][:10])


class LineEndingTests(unittest.TestCase):
    """A CRLF file written on Windows hashes differently from Render's Linux checkout."""

    HASHED_TEXT_FILES = (
        "models/v2/source_allocation_physics.json",
        "benchmarks/daily_source_allocation_v2/evaluation.json",
        "benchmarks/daily_source_allocation_v2/physics_evaluation.json",
        "data/processed/eirgrid_source_curtailment_daily.csv",
        "data/processed/eirgrid_source_curtailment_quality_report.json",
    )

    def test_hashed_text_files_are_lf_and_stored_byte_for_byte(self) -> None:
        attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
        for name in self.HASHED_TEXT_FILES:
            self.assertNotIn(b"\r", (ROOT / name).read_bytes(), name)
        for pattern in (
            "models/v2/source_allocation_physics.json -text",
            "benchmarks/daily_source_allocation_v2/*.json -text",
            "data/processed/eirgrid_source_curtailment_daily.csv -text",
            "data/processed/eirgrid_source_curtailment_quality_report.json -text",
        ):
            self.assertIn(pattern, attributes)

    def test_serving_artifact_hash_matches_its_exact_bytes(self) -> None:
        report = json.loads(PHYSICS_REPORT.read_text(encoding="utf-8"))
        import hashlib

        self.assertEqual(hashlib.sha256(PHYSICS_ARTIFACT.read_bytes()).hexdigest(), report["serving_artifact_sha256"])


class SourcePredictionApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        daily = DailyCurtailmentService()
        cls.client = TestClient(create_app(
            api_key="team-secret", daily_service=daily,
            source_allocation_service=DatasetBackedSplit(daily),
        ))
        cls.client.__enter__()
        cls.headers = {"X-API-Key": "team-secret"}

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.__exit__(None, None, None)

    def test_prediction_route_returns_experimental_reconciled_split(self) -> None:
        response = self.client.post("/predict/curtailment/sources/day", json={"target_date_utc": "2026-05-10"}, headers=self.headers)
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertTrue(body["experimental"])
        self.assertEqual(body["validation_status"], "candidate_awaiting_fresh_confirmation")
        self.assertAlmostEqual(body["predicted_wind_curtailment_mwh"] + body["predicted_solar_curtailment_mwh"], body["predicted_curtailment_mwh"], places=9)
        self.assertEqual(body["compare_with_actual"], "/actuals/curtailment/sources?target_date_utc=2026-05-10")
        self.assertAlmostEqual(body["predicted_wind_share_percent"] + body["predicted_solar_share_percent"], 100, places=1)

    def test_model_info_and_auth(self) -> None:
        self.assertEqual(self.client.get("/model-info/curtailment/sources").status_code, 401)
        info = self.client.get("/model-info/curtailment/sources", headers=self.headers).json()
        self.assertTrue(info["experimental"])
        self.assertIn("provisional", info["validation"]["provisional_2026_test"]["note"].lower())
        self.assertEqual(info["validation"]["fresh_confirmation"]["status"], "pending")

    def test_split_is_unavailable_without_v2_and_v1_split_does_not_exist(self) -> None:
        with TestClient(create_app(api_key="k")) as client:
            response = client.post("/predict/curtailment/sources/day", json={"target_date_utc": "2026-05-10"}, headers={"X-API-Key": "k"})
            self.assertEqual(response.status_code, 503)
            paths = client.get("/openapi.json").json()["paths"]
        self.assertNotIn("/predict/v1/curtailment/sources", paths)


if __name__ == "__main__":
    unittest.main()
