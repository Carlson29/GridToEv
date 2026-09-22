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

from gridtoev.inference import FeatureValidationError, PredictionService
from gridtoev.training import TrainingConfig, chronological_split, train_and_save


def make_synthetic_dataset(path: Path, periods: int = 120) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=periods, freq="30min", tz="UTC")
    rows = []
    for index, timestamp in enumerate(timestamps):
        for horizon in (30, 60):
            wind = 900 + 450 * np.sin(index / 7)
            load = 1500 + 250 * np.cos(index / 11)
            snsp = 0.4 + 0.25 * np.sin(index / 9)
            lagged_dd = max(0.0, wind - load * 0.65)
            dispatch_down = max(0.0, lagged_dd + horizon / 12 - 40)
            curtailment = dispatch_down * 0.35
            constraint = dispatch_down - curtailment
            rows.append(
                {
                    "issue_timestamp_utc": timestamp,
                    "target_timestamp_utc": timestamp + pd.Timedelta(minutes=horizon),
                    "forecast_horizon_minutes": horizon,
                    "wind_lag_1": wind,
                    "load_lag_1": load,
                    "snsp_lag_1": snsp,
                    "dispatch_down_mwh_lag_1": lagged_dd,
                    "dispatch_down_event_lag_1": int(lagged_dd > 0),
                    "dispatch_down_event": int(dispatch_down > 0),
                    "dispatch_down_mwh": dispatch_down,
                    "curtailment_mwh": curtailment,
                    "constraint_mwh": constraint,
                    "high_frequency_min_generation_mwh": curtailment,
                    "rocof_inertia_mwh": 0.0,
                    "snsp_curtailment_mwh": 0.0,
                    "transmission_constraint_mwh": constraint,
                    "tso_test_mwh": 0.0,
                    "other_reduction_mwh": 0.0,
                    "recoverable_surplus_upper_bound_mwh": dispatch_down,
                    "recoverable_surplus_100mw_flex_mwh": min(dispatch_down, 50.0),
                }
            )
    data = pd.DataFrame(rows)
    data.to_csv(path, index=False)
    return data


class TrainingAndInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.temp_path = Path(cls.temp_dir.name)
        cls.dataset_path = cls.temp_path / "dataset.csv"
        cls.artifact_path = cls.temp_path / "model.joblib"
        cls.data = make_synthetic_dataset(cls.dataset_path)
        config = TrainingConfig(max_iter=20, min_samples_leaf=5, random_state=7)
        cls.result = train_and_save(
            dataset_path=cls.dataset_path,
            artifact_path=cls.artifact_path,
            config=config,
        )
        cls.service = PredictionService(
            model_path=cls.artifact_path,
            dataset_path=cls.dataset_path,
        )
        cls.service.load()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def test_chronological_split_has_no_timestamp_overlap(self) -> None:
        train, validation, test = chronological_split(self.data)
        train_times = set(train["issue_timestamp_utc"])
        validation_times = set(validation["issue_timestamp_utc"])
        test_times = set(test["issue_timestamp_utc"])
        self.assertFalse(train_times & validation_times)
        self.assertFalse(train_times & test_times)
        self.assertFalse(validation_times & test_times)
        self.assertLess(max(train_times), min(validation_times))
        self.assertLess(max(validation_times), min(test_times))

    def test_training_writes_complete_bundle(self) -> None:
        self.assertTrue(self.artifact_path.exists())
        self.assertEqual(
            set(self.result.bundle["models"]),
            {
                "event_classifier",
                "dispatch_down_regressor",
                "curtailment_regressor",
                "constraint_regressor",
                "dispatch_down_quantile_p10",
                "dispatch_down_quantile_p50",
                "dispatch_down_quantile_p90",
            },
        )
        self.assertIn("classification", self.result.metrics)
        self.assertIn("regression", self.result.metrics)

    def test_dataset_prediction_is_bounded_and_reconciled(self) -> None:
        timestamp = self.data["issue_timestamp_utc"].iloc[-20]
        prediction = self.service.predict_from_dataset(timestamp, 30, 100.0)
        self.assertGreaterEqual(prediction["dispatch_down_probability"], 0.0)
        self.assertLessEqual(prediction["dispatch_down_probability"], 1.0)
        self.assertGreaterEqual(prediction["predicted_dispatch_down_mwh"], 0.0)
        self.assertLessEqual(
            prediction["recoverable_surplus_mwh"],
            prediction["predicted_dispatch_down_mwh"],
        )
        component_sum = (
            prediction["predicted_curtailment_mwh"]
            + prediction["predicted_constraint_mwh"]
        )
        self.assertAlmostEqual(
            component_sum,
            prediction["predicted_dispatch_down_mwh"],
            places=6,
        )
        self.assertLessEqual(prediction["prediction_interval_p10_mwh"], prediction["prediction_interval_p50_mwh"])
        self.assertLessEqual(prediction["prediction_interval_p50_mwh"], prediction["prediction_interval_p90_mwh"])

    def test_live_feature_prediction_rejects_missing_features(self) -> None:
        with self.assertRaises(FeatureValidationError):
            self.service.predict_features(
                features={"wind_lag_1": 1000.0},
                forecast_horizon_minutes=30,
                flexible_load_capacity_mw=100.0,
            )


if __name__ == "__main__":
    unittest.main()
