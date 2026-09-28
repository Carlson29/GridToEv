import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from gridtoev.api import create_app
from gridtoev.source_allocation import (
    BASELINES,
    CANDIDATES,
    add_physics_features,
    allocate,
    capacity_proxies,
    feature_columns,
    fit_share_candidates,
    load_dataset,
    predict_wind_shares,
    score,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "benchmarks" / "daily_source_allocation_v2" / "evaluation.json"
PHYSICS_COLUMNS = [
    "solar_capacity_proxy_mw", "wind_capacity_proxy_mw", "capacity_proxy_through_utc",
    "solar_energy_proxy_mwh", "wind_energy_proxy_mwh", "physics_log_ratio",
]


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

    def test_failed_experiment_is_not_served_and_only_the_physics_split_is_live(self) -> None:
        gate = self.report["release_gate"]
        self.assertFalse(gate["passed"])
        self.assertEqual(self.report["status"], "experimental_estimate_only")
        paths = {route.path for route in create_app().routes}
        # Experiment 1's allocators are never served; V1 has no split route.
        self.assertNotIn("/predict/v1/curtailment/sources", paths)
        # The V2 split is live only as the experimental physics model (owner decision).
        self.assertIn("/predict/curtailment/sources/day", paths)


class PhysicsExperimentTests(unittest.TestCase):
    def test_capacity_proxy_reads_only_published_months(self) -> None:
        history = pd.DataFrame({
            "timestamp_utc": pd.date_range("2024-01-01", "2024-12-31 23:30", freq="30min", tz="UTC"),
        })
        # Availability equals the day of year, so the proxy reveals which day it read.
        history["eirgrid_ie_solar_availability_mw"] = history["timestamp_utc"].dt.dayofyear.astype(float)
        history["eirgrid_ie_wind_availability_mw"] = 1.0
        issues = pd.Series(pd.to_datetime(["2024-09-01", "2024-09-30"], utc=True))
        proxy = capacity_proxies(history, issues)
        # Issue days in September may only use data through 31 July (month M-2).
        self.assertEqual(proxy["capacity_proxy_through_utc"].dt.strftime("%Y-%m-%d").tolist(), ["2024-07-31", "2024-07-31"])
        self.assertEqual(proxy["solar_capacity_proxy_mw"].tolist(), [213.0, 213.0])

    def test_physics_share_grows_with_forecast_wind(self) -> None:
        data = load_dataset().head(3).copy()
        history = pd.read_csv(ROOT / "data/processed/eirgrid_core_history_30min.csv.gz",
                              usecols=["timestamp_utc", "eirgrid_ie_solar_availability_mw", "eirgrid_ie_wind_availability_mw"])
        featured = add_physics_features(data, history)
        self.assertTrue(featured["physics_log_ratio"].notna().all())
        calm, windy = featured.iloc[[0]].copy(), featured.iloc[[0]].copy()
        for region_column in [column for column in featured if column.endswith("_wind_mean_kmh")]:
            calm[region_column], windy[region_column] = 15.0, 40.0
        self.assertLess(
            float(add_physics_features(calm.drop(columns=PHYSICS_COLUMNS), history)["physics_log_ratio"].iloc[0]),
            float(add_physics_features(windy.drop(columns=PHYSICS_COLUMNS), history)["physics_log_ratio"].iloc[0]),
        )

    def test_physics_report_waits_for_fresh_confirmation(self) -> None:
        report = json.loads((ROOT / "benchmarks/daily_source_allocation_v2/physics_evaluation.json").read_text(encoding="utf-8"))
        self.assertEqual(report["selected_on_validation"], "physics_share")
        gate = report["release_gate"]
        self.assertTrue(gate["test_2026"]["beats_best_baseline"])
        self.assertTrue(gate["test_2026"]["holdout_viewed_during_design"])
        self.assertEqual(gate["fresh_confirmation"]["status"], "pending")
        self.assertFalse(gate["passed"])
        self.assertEqual(report["status"], "candidate_awaiting_fresh_confirmation")
        import hashlib

        self.assertEqual(
            hashlib.sha256((ROOT / "data/processed/eirgrid_core_history_30min.csv.gz").read_bytes()).hexdigest(),
            report["inputs"]["capacity_history_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
