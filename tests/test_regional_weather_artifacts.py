import json
import re
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
BENCHMARK = ROOT / "benchmarks" / "regional_weather_signals"
WEIGHTS = ROOT / "config" / "regional_weather_capacity_weights.csv"


class RegionalWeatherArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.quality = json.loads(
            (PROCESSED / "regional_weather_quality_report.json").read_text()
        )
        cls.ablation = json.loads((BENCHMARK / "ablation_report.json").read_text())

    def test_feature_matrix_is_complete_causal_and_unique(self) -> None:
        features = pd.read_csv(
            PROCESSED / "regional_weather_features_30_60.csv",
            parse_dates=["issue_timestamp_utc", "target_timestamp_utc"],
        )
        self.assertEqual(len(features), 2867)
        self.assertEqual(len(features), self.quality["rows"])
        self.assertEqual(set(features["forecast_horizon_minutes"]), {30, 60})
        self.assertEqual(
            int(
                features.duplicated(
                    ["issue_timestamp_utc", "forecast_horizon_minutes"]
                ).sum()
            ),
            0,
        )
        expected_target = features["issue_timestamp_utc"] + pd.to_timedelta(
            features["forecast_horizon_minutes"], unit="m"
        )
        self.assertTrue((features["target_timestamp_utc"] == expected_target).all())
        for column in features.columns:
            if column.endswith(("_published_at_utc", "_observed_at_utc")):
                available = pd.to_datetime(features[column], utc=True)
                self.assertTrue(
                    (
                        available.isna()
                        | available.le(features["issue_timestamp_utc"])
                    ).all(),
                    column,
                )
        self.assertEqual(self.quality["future_publication_violations"], 0)
        self.assertEqual(self.quality["nonfinite_numeric_cells"], 0)

    def test_weights_and_missing_region_contract_are_reproducible(self) -> None:
        weights = pd.read_csv(WEIGHTS)
        self.assertEqual(len(weights), self.quality["region_count"])
        self.assertEqual(weights["region"].tolist(), self.quality["regions"])
        self.assertAlmostEqual(
            float(weights["wind_capacity_mw"].sum()),
            self.quality["wind_capacity_total_mw"],
        )
        self.assertAlmostEqual(
            float(weights["solar_capacity_mw"].sum()),
            self.quality["solar_capacity_total_mw"],
        )
        features = pd.read_csv(
            PROCESSED / "regional_weather_features_30_60.csv", nrows=1
        )
        for region in weights["region"]:
            self.assertIn(f"weather_region_{region}_missing_flag", features.columns)
        dictionary = pd.read_csv(
            PROCESSED / "regional_weather_feature_dictionary.csv"
        )
        self.assertEqual(set(dictionary["column"]), set(features.columns))

    def test_vintages_analysis_and_manifest_have_auditable_provenance(self) -> None:
        forecasts = pd.read_csv(
            PROCESSED / "regional_weather_forecast_vintages.csv.gz",
            parse_dates=[
                "run_initialized_at_utc",
                "published_at_utc",
                "valid_timestamp_utc",
            ],
        )
        analysis = pd.read_csv(
            PROCESSED / "regional_weather_analysis_proxy.csv.gz",
            parse_dates=["valid_timestamp_utc", "published_at_utc"],
        )
        self.assertEqual(len(forecasts), self.quality["forecast_vintage_rows"])
        self.assertEqual(len(analysis), self.quality["analysis_proxy_rows"])
        self.assertTrue(
            (
                forecasts["published_at_utc"]
                == forecasts["run_initialized_at_utc"] + pd.Timedelta(hours=6)
            ).all()
        )
        self.assertTrue(
            (
                analysis["published_at_utc"]
                == analysis["valid_timestamp_utc"] + pd.Timedelta(hours=6)
            ).all()
        )

        manifest = pd.read_csv(PROCESSED / "regional_weather_source_manifest.csv")
        self.assertEqual(len(manifest), self.quality["model_run_count"] + 1)
        self.assertNotIn("cache_hit", manifest.columns)
        self.assertTrue(
            manifest["sha256"].map(
                lambda value: bool(re.fullmatch(r"[0-9a-f]{64}", value))
            ).all()
        )
        self.assertTrue(
            manifest["url"].str.startswith(
                (
                    "https://single-runs-api.open-meteo.com/",
                    "https://historical-forecast-api.open-meteo.com/",
                )
            ).all()
        )

    def test_named_ablation_enforces_gate_without_final_test_access(self) -> None:
        self.assertEqual(
            self.ablation["experiment_name"], "regional_weather_signals"
        )
        self.assertEqual(self.ablation["overlap_rows"], 2867)
        self.assertEqual(self.ablation["development_fold_count"], 4)
        self.assertFalse(self.ablation["final_test_accessed"])
        improvements = [
            fold["mae_improvement_fraction"]
            for fold in self.ablation["metrics"]["folds"]
        ]
        passes = (
            self.ablation["metrics"]["mean_mae_improvement_fraction"] >= 0.15
            and min(improvements) >= 0
        )
        self.assertEqual(
            self.ablation["production_decision"],
            "include" if passes else "exclude",
        )


if __name__ == "__main__":
    unittest.main()
