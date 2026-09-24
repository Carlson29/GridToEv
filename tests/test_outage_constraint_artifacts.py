import json
import re
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
BENCHMARK = ROOT / "benchmarks" / "outage_constraint_signals"


class OutageConstraintArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.quality = json.loads(
            (PROCESSED / "outage_constraint_quality_report.json").read_text()
        )
        cls.ablation = json.loads((BENCHMARK / "ablation_report.json").read_text())

    def test_feature_matrix_has_one_causal_row_per_modelling_key(self) -> None:
        features = pd.read_csv(
            PROCESSED / "outage_constraint_features_30_60.csv",
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
            if column.endswith("_published_at_utc"):
                published = pd.to_datetime(features[column], utc=True)
                self.assertTrue(
                    (
                        published.isna()
                        | published.le(features["issue_timestamp_utc"])
                    ).all(),
                    column,
                )

    def test_normalized_sources_and_dictionary_are_complete(self) -> None:
        outages = pd.read_csv(PROCESSED / "outage_events.csv.gz")
        constraints = pd.read_csv(PROCESSED / "ecp_constraint_pressure.csv.gz")
        self.assertEqual(len(outages), self.quality["outage_event_rows"])
        self.assertEqual(len(constraints), self.quality["ecp_constraint_rows"])
        self.assertEqual(
            set(outages["source_family"]),
            {"generation_plan", "transmission_programme"},
        )
        self.assertTrue(
            {
                "constraint_group",
                "study_year",
                "installed_mw",
                "constraint_ratio",
                "published_at_utc",
            }.issubset(constraints.columns)
        )

        features = pd.read_csv(
            PROCESSED / "outage_constraint_features_30_60.csv", nrows=1
        )
        dictionary = pd.read_csv(
            PROCESSED / "outage_constraint_feature_dictionary.csv"
        )
        self.assertEqual(set(dictionary["column"]), set(features.columns))
        self.assertTrue(
            {"unit", "source", "formula", "availability_rule"}.issubset(
                dictionary.columns
            )
        )

    def test_manifest_records_official_urls_checksums_and_publication_times(self) -> None:
        manifest = pd.read_csv(
            PROCESSED / "outage_constraint_source_manifest.csv",
            parse_dates=["retrieved_at_utc", "published_at_utc"],
        )
        self.assertEqual(len(manifest), 3)
        self.assertEqual(set(manifest["provider"]), {"EirGrid"})
        self.assertEqual(
            set(manifest["report"]),
            {
                "all_island_generation_outage_plan",
                "transmission_outage_programme",
                "ecp_constraint_forecast",
            },
        )
        self.assertTrue(
            manifest["url"].str.startswith("https://cms.eirgrid.ie/").all()
        )
        self.assertTrue(
            manifest["sha256"].map(
                lambda value: bool(re.fullmatch(r"[0-9a-f]{64}", value))
            ).all()
        )
        self.assertTrue(
            (manifest["published_at_utc"] <= manifest["retrieved_at_utc"]).all()
        )
        self.assertNotIn("cache_hit", manifest.columns)

    def test_ablation_is_named_and_keeps_the_final_test_sealed(self) -> None:
        self.assertEqual(
            self.ablation["experiment_name"], "outage_constraint_signals"
        )
        self.assertEqual(self.ablation["overlap_rows"], 2867)
        self.assertEqual(self.ablation["development_fold_count"], 4)
        self.assertEqual(self.ablation["status"], "rejected_development_gate")
        self.assertEqual(self.ablation["production_decision"], "exclude")
        self.assertFalse(self.ablation["final_test_accessed"])
        self.assertEqual(len(self.ablation["metrics"]["folds"]), 4)


if __name__ == "__main__":
    unittest.main()
