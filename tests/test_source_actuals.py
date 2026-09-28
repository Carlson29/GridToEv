import hashlib
import json
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

from gridtoev.api import create_app
from gridtoev.source_actuals import SourceActualsService


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "tests" / "fixtures" / "openapi_existing_routes.sha256.json"
NEW_OPERATIONS = {"GET /actuals/curtailment/sources", "GET /actuals/curtailment/sources/coverage"}


def _archive(path: Path) -> None:
    """Day 1: wind only (pre-solar). Day 2: complete. Day 3: zero. Day 4: 10 half-hours only."""
    stamps = pd.date_range("2023-03-30", periods=48 * 3 + 10, freq="30min", tz="UTC")
    wind = np.r_[np.full(48, 2.0), np.full(48, 3.0), np.zeros(48), np.ones(10)]
    solar = np.r_[np.full(48, np.nan), np.full(48, 1.0), np.zeros(48), np.ones(10)]
    frame = pd.DataFrame({
        "timestamp_utc": stamps,
        "wind_curtailment_mwh": wind,
        "solar_curtailment_mwh": solar,
        "wind_reported_flag": 1,
        "solar_reported_flag": (~np.isnan(solar)).astype(int),
    })
    frame["source_labels_complete_flag"] = frame["solar_reported_flag"]
    frame.to_csv(path, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")


class SourceActualsServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        path = Path(self.temp.name) / "sources.csv"
        _archive(path)
        self.service = SourceActualsService(path)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_complete_day_returns_split_shares_and_summary(self) -> None:
        result = self.service.day(date(2023, 3, 31))
        self.assertEqual(result["status"], "available")
        self.assertEqual((result["wind_curtailment_mwh"], result["solar_curtailment_mwh"]), (144.0, 48.0))
        self.assertEqual(result["total_curtailment_mwh"], 192.0)
        self.assertEqual((result["wind_share_percent"], result["solar_share_percent"]), (75.0, 25.0))
        self.assertIn("75% wind", result["summary"])

    def test_pre_solar_day_keeps_solar_unknown_not_zero(self) -> None:
        result = self.service.day(date(2023, 3, 30))
        self.assertEqual(result["status"], "solar_not_published")
        self.assertEqual(result["wind_curtailment_mwh"], 96.0)
        self.assertIsNone(result["solar_curtailment_mwh"])
        self.assertIsNone(result["total_curtailment_mwh"])

    def test_zero_day_has_no_percentages(self) -> None:
        result = self.service.day(date(2023, 4, 1))
        self.assertEqual(result["total_curtailment_mwh"], 0.0)
        self.assertIsNone(result["solar_share_percent"])
        self.assertIn("No wind or solar power was curtailed", result["summary"])

    def test_partial_latest_day_is_pending_and_earlier_gap_is_missing(self) -> None:
        self.assertEqual(self.service.day(date(2023, 4, 2))["status"], "pending")
        self.assertEqual(self.service.day(date(2030, 1, 1))["status"], "pending")
        self.assertEqual(self.service.day(date(2020, 1, 1))["status"], "missing")

    def test_half_hours_are_optional_and_keep_nulls(self) -> None:
        self.assertNotIn("half_hours", self.service.day(date(2023, 3, 30)))
        rows = self.service.day(date(2023, 3, 30), include_half_hours=True)["half_hours"]
        self.assertEqual(len(rows), 48)
        self.assertIsNone(rows[0]["solar_curtailment_mwh"])

    def test_coverage_reports_solar_start_and_complete_days(self) -> None:
        coverage = self.service.coverage()
        self.assertEqual(coverage["solar_first_published_utc"], "2023-03-31T00:00:00+00:00")
        self.assertEqual((coverage["complete_day_min_utc"], coverage["complete_day_max_utc"]), ("2023-03-31", "2023-04-01"))


class SourceActualsApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(create_app(api_key="team-secret"))
        cls.client.__enter__()
        cls.headers = {"X-API-Key": "team-secret"}

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.__exit__(None, None, None)

    def test_routes_require_key_and_validate_dates(self) -> None:
        self.assertEqual(self.client.get("/actuals/curtailment/sources?target_date_utc=2026-05-10").status_code, 401)
        self.assertEqual(self.client.get("/actuals/curtailment/sources?target_date_utc=10-05-2026", headers=self.headers).status_code, 422)
        self.assertEqual(self.client.get("/actuals/curtailment/sources/coverage", headers=self.headers).status_code, 200)

    def test_source_total_matches_existing_daily_actual(self) -> None:
        for day in ("2024-07-01", "2025-03-14", "2026-05-10"):
            sources = self.client.get(f"/actuals/curtailment/sources?target_date_utc={day}", headers=self.headers).json()
            daily = self.client.get(f"/actuals/daily-curtailment?target_date_utc={day}", headers=self.headers).json()
            self.assertEqual(sources["status"], "available")
            self.assertAlmostEqual(sources["total_curtailment_mwh"], daily["actual_curtailment_mwh"], places=6)
            self.assertAlmostEqual(sources["wind_curtailment_mwh"] + sources["solar_curtailment_mwh"], sources["total_curtailment_mwh"], places=9)

    def test_existing_routes_and_schemas_are_byte_for_byte_unchanged(self) -> None:
        schema = self.client.get("/openapi.json").json()
        snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))

        def digest(value: object) -> str:
            return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

        operations = {f"{method.upper()} {path}": op for path, ops in schema["paths"].items() for method, op in ops.items()}
        for key, expected in snapshot["paths"].items():
            self.assertEqual(digest(operations[key]), expected, key)
        components = schema["components"]["schemas"]
        for name, expected in snapshot["components"].items():
            self.assertEqual(digest(components[name]), expected, name)
        self.assertEqual(digest(schema["info"]["description"]), snapshot["info_description"])
        self.assertEqual(set(operations) - set(snapshot["paths"]), NEW_OPERATIONS)


class SourceActualsDeploymentTests(unittest.TestCase):
    def test_docker_image_ships_the_source_archive(self) -> None:
        archive = "data/processed/eirgrid_source_curtailment_30min.csv.gz"
        self.assertIn(f"!{archive}", (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines())
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn(f"COPY --chown=gridtoev:gridtoev {archive} ./{archive}", dockerfile)
        self.assertIn(f"GRIDTOEV_SOURCE_ACTUALS_PATH=/app/{archive}", dockerfile)


if __name__ == "__main__":
    unittest.main()
