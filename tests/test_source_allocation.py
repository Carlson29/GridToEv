import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from gridtoev.api import create_app
from gridtoev.source_allocation import (
    BASELINES,
    CANDIDATES,
    allocate,
    feature_columns,
    fit_share_candidates,
    load_dataset,
    predict_wind_shares,
    score,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "benchmarks" / "daily_source_allocation_v2" / "evaluation.json"
SOURCE_ROUTES = {"/predict/v1/curtailment/sources", "/predict/curtailment/sources/day"}


class AllocationTests(unittest.TestCase):
    def test_zero_parent_gives_zero_components(self) -> None:
        wind, solar = allocate(np.array([0.0, 0.0]), np.array([0.3, 1.0]))
        self.assertEqual(wind.tolist(), [0.0, 0.0])
        self.assertEqual(solar.tolist(), [0.0, 0.0])

    def test_positive_parent_is_conserved_and_non_negative(self) -> None:
        total = np.array([100.0, 55.5, 7.0])
        wind, solar = allocate(total, np.array([0.8, 1.4, -0.2]))  # out-of-range shares are clamped
        np.testing.assert_allclose(wind + solar, total)
        self.assertTrue((wind >= 0).all() and (solar >= 0).all())
        self.assertEqual(wind.tolist(), [80.0, 55.5, 0.0])

    def test_score_reports_conservation_and_solar_active_slice(self) -> None:
        frame = pd.DataFrame({"wind_curtailment_mwh": [90.0, 0.0], "solar_curtailment_mwh": [10.0, 0.0]})
        result = score(frame, np.array([100.0, 0.0]), np.array([0.9, 0.5]))
        self.assertAlmostEqual(result["combined_source_mae_mwh"], 0.0)
        self.assertEqual(result["solar_active_days"], 1)
        self.assertLess(result["max_conservation_error_mwh"], 1e-9)


class AllocationDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = load_dataset()

    def test_labels_are_not_features(self) -> None:
        columns = feature_columns(self.data)
        for label in ("curtailment_mwh", "curtailment_event", "wind_curtailment_mwh", "solar_curtailment_mwh",
                      "source_curtailment_total_mwh", "wind_share"):
            self.assertNotIn(label, columns)
        self.assertTrue((pd.to_datetime(self.data["forecast_max_available_at_utc"], utc=True)
                         <= self.data["issue_timestamp_utc"]).all())

    def test_every_candidate_returns_bounded_shares(self) -> None:
        columns = feature_columns(self.data)
        fit = self.data.loc[self.data["issue_timestamp_utc"].lt("2025-01-01T00:00:00Z")]
        score_rows = self.data.loc[self.data["issue_timestamp_utc"].ge("2026-01-01T00:00:00Z")]
        shares = predict_wind_shares(fit_share_candidates(fit, columns), score_rows, columns)
        self.assertEqual(set(shares), set(CANDIDATES))
        for value in shares.values():
            self.assertTrue(((value >= 0) & (value <= 1)).all())


class AllocationReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.report = json.loads(REPORT.read_text(encoding="utf-8"))

    def test_partitions_are_chronological_and_disjoint(self) -> None:
        ranges = self.report["date_ranges"]
        self.assertLess(ranges["train"][1], ranges["validation"][0])
        self.assertLess(ranges["validation"][1], ranges["test"][0])
        self.assertEqual(self.report["rows"], {"train": 275, "validation": 365, "test": 242})

    def test_report_matches_committed_inputs(self) -> None:
        import hashlib

        inputs = self.report["inputs"]
        for key, path in (
            ("dataset_sha256", "data/processed/daily_curtailment_forecast_v2.csv.gz"),
            ("source_labels_sha256", "data/processed/eirgrid_source_curtailment_daily.csv"),
            ("parent_artifact_sha256", "models/v2/daily_curtailment_bundle.joblib"),
        ):
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), inputs[key], path)

    def test_all_candidates_conserve_the_predicted_parent_total(self) -> None:
        for name in CANDIDATES:
            self.assertLess(self.report["test_end_to_end"][name]["max_conservation_error_mwh"], 1e-6)
        self.assertTrue(set(BASELINES) <= set(CANDIDATES))

    def test_unvalidated_source_routes_stay_unavailable(self) -> None:
        gate = self.report["release_gate"]
        self.assertFalse(gate["passed"])
        self.assertEqual(self.report["status"], "experimental_estimate_only")
        paths = {route.path for route in create_app().routes}
        # The handoff forbids serving an allocator that has not passed its gate.
        self.assertFalse(SOURCE_ROUTES & paths)


if __name__ == "__main__":
    unittest.main()
