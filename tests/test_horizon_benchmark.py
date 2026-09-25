import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gridtoev.horizon_benchmark import (
    CandidateSpec,
    HorizonSpecificModel,
    assess_release_candidate,
    default_candidate_specs,
    evaluate_prediction_set,
    feature_family_columns,
    select_horizon_ensemble_weights,
)


def make_frame(periods: int = 80) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    timestamps = pd.date_range("2026-01-01", periods=periods, freq="30min", tz="UTC")
    for index, timestamp in enumerate(timestamps):
        latest = 10.0 + index / 8.0
        for horizon in (30, 60):
            target = latest + (2.0 if horizon == 30 else 7.0)
            rows.append(
                {
                    "issue_timestamp_utc": timestamp,
                    "target_timestamp_utc": timestamp + pd.Timedelta(minutes=horizon),
                    "forecast_horizon_minutes": horizon,
                    "dispatch_down_mwh_latest_observed": latest,
                    "dispatch_down_mwh_lag_1": latest - 1.0,
                    "dispatch_down_event": 1,
                    "dispatch_down_mwh": target,
                    "wind_lag_1": 1000.0 + index,
                    "eirgrid_snsp_ratio": 0.5 + index / 1000.0,
                    "entsoe_price_eur_mwh": 80.0 + index / 10.0,
                    "hour_sin": np.sin(index / 12.0),
                }
            )
    return pd.DataFrame(rows)


class HorizonBenchmarkTests(unittest.TestCase):
    def test_required_standard_candidates_are_present(self) -> None:
        names = {candidate.name for candidate in default_candidate_specs()}
        self.assertIn("ridge_residual", names)
        self.assertIn("hist_gradient_boosting_residual", names)
        self.assertIn("extra_trees_residual", names)
        self.assertIn("two_stage_event_quantity", names)

    def test_horizon_model_fits_separate_models_and_clips_predictions(self) -> None:
        frame = make_frame()
        model = HorizonSpecificModel(
            CandidateSpec("ridge", "ridge_residual", {"alpha": 1.0}),
            ["forecast_horizon_minutes", "wind_lag_1", "dispatch_down_mwh_lag_1"],
            random_state=7,
        )
        model.fit(frame)
        predictions = model.predict(frame)

        self.assertEqual(set(model.models), {30, 60})
        self.assertEqual(len(predictions), len(frame))
        self.assertTrue(np.isfinite(predictions).all())
        self.assertTrue((predictions >= 0).all())

    def test_prediction_report_contains_fold_horizon_and_regime_metrics(self) -> None:
        frame = make_frame(periods=12)
        predictions = frame["dispatch_down_mwh"].to_numpy(dtype=float) + 1.0
        folds = np.repeat(["fold_1", "fold_2", "fold_3"], repeats=len(frame) // 3)
        report = evaluate_prediction_set(frame, predictions, folds)

        self.assertAlmostEqual(report["overall"]["mae"], 1.0)
        self.assertEqual(set(report["by_horizon"]), {"30", "60"})
        self.assertIn("positive_dispatch_down", report["by_event_regime"])
        self.assertEqual(len(report["folds"]), 3)
        self.assertIn("fold_mae_std_mwh", report["dispersion"])

    def test_release_gate_requires_aggregate_horizon_and_fold_stability(self) -> None:
        baseline = {
            "overall": {"mae": 10.0},
            "by_horizon": {"30": {"mae": 10.0}, "60": {"mae": 10.0}},
            "folds": [{"name": f"fold_{index}", "overall": {"mae": 10.0}} for index in range(4)],
        }
        passing = {
            "overall": {"mae": 8.0},
            "by_horizon": {"30": {"mae": 8.0}, "60": {"mae": 8.0}},
            "folds": [{"name": f"fold_{index}", "overall": {"mae": 8.0}} for index in range(4)],
        }
        one_fold_win = {
            "overall": {"mae": 8.0},
            "by_horizon": {"30": {"mae": 8.0}, "60": {"mae": 8.0}},
            "folds": [
                {"name": "fold_0", "overall": {"mae": 2.0}},
                {"name": "fold_1", "overall": {"mae": 10.5}},
                {"name": "fold_2", "overall": {"mae": 10.5}},
                {"name": "fold_3", "overall": {"mae": 9.0}},
            ],
        }
        thresholds = {"rolling_mae_improvement_min": 0.15}

        self.assertTrue(assess_release_candidate(passing, baseline, thresholds)["passed"])
        rejected = assess_release_candidate(one_fold_win, baseline, thresholds)
        self.assertFalse(rejected["passed"])
        self.assertIn("fold_stability", rejected["failed_checks"])

    def test_ensemble_weights_are_selected_independently_by_horizon(self) -> None:
        frame = make_frame(periods=20)
        truth = frame["dispatch_down_mwh"].to_numpy(dtype=float)
        horizons = frame["forecast_horizon_minutes"].to_numpy()
        first = truth.copy()
        second = truth.copy()
        first[horizons == 60] += 8.0
        second[horizons == 30] += 8.0
        folds = np.where(frame["issue_timestamp_utc"] < frame["issue_timestamp_utc"].median(), "a", "b")

        weights, predictions = select_horizon_ensemble_weights(
            frame,
            folds,
            {"first": first, "second": second},
            ("first", "second"),
        )

        self.assertEqual(weights["30"]["first"], 1.0)
        self.assertEqual(weights["60"]["second"], 1.0)
        self.assertAlmostEqual(float(np.abs(predictions - truth).max()), 0.0)

    def test_feature_families_are_nonempty_and_subset_the_contract(self) -> None:
        columns = [
            "forecast_horizon_minutes",
            "dispatch_down_mwh_latest_observed",
            "dispatch_down_mwh_lag_1",
            "wind_lag_1",
            "eirgrid_snsp_ratio",
            "entsoe_price_eur_mwh",
            "hour_sin",
        ]
        groups = feature_family_columns(columns)

        self.assertIn("all_features", groups)
        self.assertIn("persistence_history", groups)
        self.assertIn("renewable_grid_state", groups)
        self.assertTrue(all(set(group).issubset(columns) for group in groups.values()))
        self.assertTrue(all(group for group in groups.values()))


if __name__ == "__main__":
    unittest.main()
