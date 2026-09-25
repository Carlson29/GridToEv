import json
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "benchmarks" / "horizon_models"


class HorizonBenchmarkArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads(
            (BENCHMARK_DIR / "benchmark_report.json").read_text(encoding="utf-8")
        )
        cls.registry = pd.read_csv(BENCHMARK_DIR / "experiment_registry.csv")
        cls.predictions = pd.read_csv(
            BENCHMARK_DIR / "development_oof_predictions.csv.gz"
        )

    def test_report_uses_frozen_contract_and_required_baselines(self) -> None:
        self.assertEqual(
            self.report["contract"]["contract_id"], "dispatch-down-benchmark-v1"
        )
        self.assertEqual(
            set(self.report["baselines"]),
            {"v1.1.0", "latest_observation", "stale_persistence"},
        )

    def test_all_mandated_candidate_families_were_run(self) -> None:
        candidates = self.report["candidates"]
        self.assertIn("ridge_residual", candidates)
        self.assertIn("hist_gradient_boosting_residual", candidates)
        self.assertIn("extra_trees_residual", candidates)
        self.assertIn("two_stage_event_quantity", candidates)
        self.assertTrue(any(name.startswith("ensemble__") for name in candidates))
        self.assertTrue(any("v1.1.0" in name for name in candidates if name.startswith("ensemble__")))

    def test_negative_release_decision_keeps_final_test_sealed(self) -> None:
        selection = self.report["selection"]
        self.assertEqual(selection["production_decision"], "retain_v1.1.0")
        self.assertEqual(selection["active_model"], "v1.1.0")
        self.assertFalse(selection["development_gate"]["passed"])
        self.assertFalse(self.report["final_test"]["accessed"])
        self.assertIsNone(self.report["final_test"]["metrics"])

    def test_oof_predictions_are_complete_and_naturally_unique(self) -> None:
        self.assertFalse(
            self.predictions.duplicated(
                ["issue_timestamp_utc", "forecast_horizon_minutes", "fold"]
            ).any()
        )
        self.assertFalse(self.predictions.isna().any().any())
        self.assertEqual(
            set(self.predictions["forecast_horizon_minutes"].astype(int)), {30, 60}
        )

    def test_registry_and_ablation_outputs_are_complete(self) -> None:
        self.assertGreaterEqual(len(self.registry), 15)
        for column in (
            "rolling_mae_mwh",
            "rolling_30m_mae_mwh",
            "rolling_60m_mae_mwh",
            "fold_mae_std_mwh",
            "training_seconds",
            "inference_seconds",
        ):
            self.assertIn(column, self.registry.columns)
        ablations = self.report["feature_family_ablations"]["results"]
        self.assertIn("all_features", ablations)
        self.assertIn("persistence_history", ablations)
        self.assertIn("renewable_grid_state", ablations)


if __name__ == "__main__":
    unittest.main()
