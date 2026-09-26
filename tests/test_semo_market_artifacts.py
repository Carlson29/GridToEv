import json
import re
import unittest
from pathlib import Path

import pandas as pd

from gridtoev.semo_market import ACTUAL_OUTCOME_DATASETS, SUPPORTED_DATASETS


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
BENCHMARK = ROOT / "benchmarks" / "semo_market_signals"


class SemoMarketArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.quality = json.loads(
            (PROCESSED / "semo_market_quality_report.json").read_text()
        )
        cls.ablation = json.loads(
            (BENCHMARK / "ablation_report.json").read_text()
        )

    def test_normalized_signals_are_causal_and_auditable(self) -> None:
        signals = pd.read_csv(
            PROCESSED / "semo_market_signals.csv.gz",
            low_memory=False,
            parse_dates=[
                "target_end_timestamp_utc",
                "published_at_utc",
                "downloaded_at_utc",
            ],
        )
        self.assertEqual(len(signals), self.quality["signal_rows"])
        self.assertTrue(set(signals["dataset_name"]).issubset(SUPPORTED_DATASETS))
        self.assertTrue(
            (signals["published_at_utc"] <= signals["downloaded_at_utc"]).all()
        )

        actual = signals["dataset_name"].isin(ACTUAL_OUTCOME_DATASETS)
        interval_known = signals["target_end_timestamp_utc"].notna()
        self.assertTrue(
            (
                signals.loc[actual & interval_known, "published_at_utc"]
                >= signals.loc[actual & interval_known, "target_end_timestamp_utc"]
            ).all()
        )

    def test_feature_matrix_has_no_future_information(self) -> None:
        features = pd.read_csv(
            PROCESSED / "semo_market_features_asof_30_60.csv",
            parse_dates=["issue_timestamp_utc", "target_timestamp_utc"],
        )
        self.assertEqual(len(features), self.quality["feature_rows"])
        self.assertEqual(set(features["forecast_horizon_minutes"]), {30, 60})
        expected_target = features["issue_timestamp_utc"] + pd.to_timedelta(
            features["forecast_horizon_minutes"], unit="m"
        )
        self.assertTrue((features["target_timestamp_utc"] == expected_target).all())
        self.assertEqual(
            int(
                features.duplicated(
                    ["issue_timestamp_utc", "forecast_horizon_minutes"]
                ).sum()
            ),
            0,
        )
        for column in features.columns:
            if column.endswith(("_published_at_utc", "_observed_at_utc")):
                timestamp = pd.to_datetime(features[column], utc=True)
                self.assertTrue(
                    (timestamp.isna() | (timestamp <= features["issue_timestamp_utc"])).all(),
                    column,
                )

    def test_dictionary_manifest_and_quality_cover_the_artifacts(self) -> None:
        features = pd.read_csv(
            PROCESSED / "semo_market_features_asof_30_60.csv", nrows=1
        )
        dictionary = pd.read_csv(PROCESSED / "semo_market_feature_dictionary.csv")
        self.assertEqual(set(dictionary["column"]), set(features.columns))

        manifest = pd.read_csv(PROCESSED / "semo_market_source_manifest.csv")
        required = {
            "provider",
            "report",
            "url",
            "retrieved_at_utc",
            "sha256",
            "schema_version",
            "first_interval_utc",
            "last_interval_utc",
            "row_count",
        }
        self.assertTrue(required.issubset(manifest.columns))
        self.assertNotIn("cache_hit", manifest.columns)
        valid_hashes = manifest["sha256"].map(
            lambda value: bool(re.fullmatch(r"[0-9a-f]{64}", value))
        )
        self.assertTrue(valid_hashes.all())
        self.assertEqual(set(self.quality["report_coverage"]), set(SUPPORTED_DATASETS))
        self.assertEqual(self.quality["future_publication_violations"], 0)
        self.assertEqual(self.quality["future_observation_violations"], 0)

    def test_named_ablation_is_recorded_without_using_the_final_test(self) -> None:
        self.assertEqual(
            self.ablation["experiment_name"], "semo_market_operational_signals"
        )
        self.assertEqual(self.ablation["status"], "rejected_no_temporal_overlap")
        self.assertEqual(self.ablation["production_decision"], "exclude")
        self.assertEqual(self.ablation["overlap_rows"], 0)
        self.assertIsNone(self.ablation["metrics"])
        self.assertFalse(self.ablation["final_test_accessed"])


if __name__ == "__main__":
    unittest.main()
