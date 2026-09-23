import hashlib
import inspect
import json
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

from gridtoev.benchmarking import (
    BenchmarkDataError,
    build_rolling_origin_folds,
    load_benchmark_contract,
    run_baseline_benchmark,
    validate_benchmark_dataset,
)
from gridtoev.training import (
    TrainingConfig,
    _select_operating_policy,
    chronological_split,
    load_dataset,
)


def make_benchmark_dataset(path: Path, periods: int = 120) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=periods, freq="30min", tz="UTC")
    rows: list[dict[str, object]] = []
    for index, timestamp in enumerate(timestamps):
        for horizon in (30, 60):
            wind = 900 + 450 * np.sin(index / 7)
            load = 1500 + 250 * np.cos(index / 11)
            lagged = max(0.0, wind - load * 0.65)
            latest = lagged + 5.0
            dispatch_down = max(0.0, lagged + horizon / 12 - 40)
            curtailment = dispatch_down * 0.35
            constraint = dispatch_down - curtailment
            rows.append(
                {
                    "issue_timestamp_utc": timestamp,
                    "target_timestamp_utc": timestamp + pd.Timedelta(minutes=horizon),
                    "forecast_horizon_minutes": horizon,
                    "wind_lag_1": wind,
                    "load_lag_1": load,
                    "dispatch_down_mwh_lag_1": lagged,
                    "dispatch_down_mwh_latest_observed": latest,
                    "dispatch_down_event_latest_observed": int(latest > 0),
                    "dispatch_down_event_lag_1": int(lagged > 0),
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


class BenchmarkingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.dataset_path = self.temp_path / "dataset.csv"
        self.data = make_benchmark_dataset(self.dataset_path)
        self.contract_path = ROOT / "config" / "benchmark_contract.v1.json"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_contract_versions_folds_and_release_thresholds(self) -> None:
        contract = load_benchmark_contract(self.contract_path)

        self.assertEqual(contract.contract_id, "dispatch-down-benchmark-v1")
        self.assertEqual(contract.baseline_model_version, "1.1.0")
        self.assertEqual(
            contract.rolling_origin_windows,
            ((0.40, 0.60), (0.60, 0.80), (0.80, 1.00)),
        )
        self.assertEqual(contract.release_thresholds["final_test_mae_mwh_max"], 9.5)
        self.assertEqual(contract.release_thresholds["rolling_mae_improvement_min"], 0.15)

    def test_shuffled_dataset_fails_before_benchmarking(self) -> None:
        shuffled_path = self.temp_path / "shuffled.csv"
        self.data.sample(frac=1.0, random_state=4).to_csv(shuffled_path, index=False)

        with self.assertRaisesRegex(BenchmarkDataError, "chronologically sorted"):
            validate_benchmark_dataset(shuffled_path)

    def test_future_published_feature_fails_leakage_gate(self) -> None:
        leaking_path = self.temp_path / "future-published.csv"
        leaking = self.data.copy()
        leaking["wind_forecast_published_at_utc"] = leaking["issue_timestamp_utc"]
        leaking.loc[0, "wind_forecast_published_at_utc"] = (
            leaking.loc[0, "issue_timestamp_utc"] + pd.Timedelta(minutes=5)
        )
        leaking.to_csv(leaking_path, index=False)

        with self.assertRaisesRegex(BenchmarkDataError, "published after issue_timestamp_utc"):
            validate_benchmark_dataset(leaking_path)

    def test_target_misalignment_fails_leakage_gate(self) -> None:
        leaking_path = self.temp_path / "misaligned-target.csv"
        leaking = self.data.copy()
        leaking.loc[0, "target_timestamp_utc"] = leaking.loc[0, "issue_timestamp_utc"]
        leaking.to_csv(leaking_path, index=False)

        with self.assertRaisesRegex(BenchmarkDataError, "strictly future"):
            validate_benchmark_dataset(leaking_path)

    def test_rolling_folds_are_expanding_and_never_touch_final_test(self) -> None:
        contract = load_benchmark_contract(self.contract_path)
        validate_benchmark_dataset(self.dataset_path)
        data = load_dataset(self.dataset_path)
        train, validation, final_test = chronological_split(
            data,
            train_fraction=contract.train_fraction,
            validation_fraction=contract.validation_fraction,
        )
        folds = build_rolling_origin_folds(train, validation, contract)

        self.assertEqual(len(folds), 4)
        previous_fit_rows = 0
        for fold in folds:
            self.assertGreater(len(fold.fit), previous_fit_rows)
            self.assertLess(
                fold.fit["issue_timestamp_utc"].max(),
                fold.score["issue_timestamp_utc"].min(),
            )
            self.assertLess(
                fold.score["issue_timestamp_utc"].max(),
                final_test["issue_timestamp_utc"].min(),
            )
            previous_fit_rows = len(fold.fit)

    def test_policy_selection_interface_cannot_receive_final_test(self) -> None:
        parameters = inspect.signature(_select_operating_policy).parameters
        self.assertNotIn("test", parameters)
        self.assertNotIn("final_test", parameters)

    def test_benchmark_writes_granular_report_and_experiment_registry(self) -> None:
        output_dir = self.temp_path / "benchmark"
        rollback_artifact = ROOT / "models" / "gridtoev_model_bundle.joblib"
        artifact_hash_before = hashlib.sha256(rollback_artifact.read_bytes()).hexdigest()
        result = run_baseline_benchmark(
            dataset_path=self.dataset_path,
            contract_path=self.contract_path,
            output_dir=output_dir,
            training_config=TrainingConfig(
                max_iter=10,
                min_samples_leaf=5,
                random_state=7,
            ),
            verify_frozen_baseline=False,
        )
        artifact_hash_after = hashlib.sha256(rollback_artifact.read_bytes()).hexdigest()

        self.assertTrue(result.report_path.exists())
        self.assertTrue(result.registry_path.exists())
        self.assertEqual(artifact_hash_before, artifact_hash_after)
        report = json.loads(result.report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["contract"]["contract_id"], "dispatch-down-benchmark-v1")
        self.assertTrue(report["final_test"]["sealed_during_selection"])
        self.assertIn("by_horizon", report["final_test"]["dispatch_breakdown"])
        self.assertIn("by_event_regime", report["final_test"]["dispatch_breakdown"])
        self.assertEqual(len(report["rolling_origin"]["folds"]), 4)
        self.assertIn("aggregate", report["rolling_origin"])

        metrics = report["final_test"]["metrics"]
        self.assertIn("average_precision", metrics["classification"]["test"])
        self.assertIn("mae", metrics["regression"]["dispatch_down_mwh"])
        self.assertIn("rmse", metrics["regression"]["dispatch_down_mwh"])
        self.assertIn("wape", metrics["regression"]["dispatch_down_mwh"])
        self.assertIn("p50_pinball_loss", metrics["uncertainty"])
        self.assertIn("p10_p90_empirical_coverage", metrics["uncertainty"])

        registry = pd.read_csv(result.registry_path)
        self.assertEqual(len(registry), 1)
        for column in (
            "contract_id",
            "dataset_sha256",
            "feature_groups",
            "settings_json",
            "runtime_seconds",
            "final_test_mae_mwh",
            "final_test_wape",
            "final_test_average_precision",
        ):
            self.assertIn(column, registry.columns)

    def test_benchmark_registry_preserves_other_experiment_rows(self) -> None:
        output_dir = self.temp_path / "benchmark-preserve"
        output_dir.mkdir()
        registry_path = output_dir / "experiment_registry.csv"
        pd.DataFrame(
            [
                {
                    "run_id": "candidate-run",
                    "contract_id": "dispatch-down-benchmark-v1",
                    "status": "candidate",
                }
            ]
        ).to_csv(registry_path, index=False)

        run_baseline_benchmark(
            dataset_path=self.dataset_path,
            contract_path=self.contract_path,
            output_dir=output_dir,
            training_config=TrainingConfig(
                max_iter=5,
                min_samples_leaf=5,
                random_state=7,
            ),
            verify_frozen_baseline=False,
        )

        registry = pd.read_csv(registry_path)
        self.assertEqual(len(registry), 2)
        self.assertEqual(int(registry["run_id"].eq("candidate-run").sum()), 1)
        self.assertEqual(int(registry["status"].eq("frozen_baseline").sum()), 1)


if __name__ == "__main__":
    unittest.main()
