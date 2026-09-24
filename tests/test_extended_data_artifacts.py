import json
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


class ExtendedDataArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.history_quality = json.loads(
            (PROCESSED / "eirgrid_history_quality_report.json").read_text()
        )
        cls.forecast_quality = json.loads(
            (PROCESSED / "forecast_vintage_quality_report.json").read_text()
        )
        cls.asof_quality = json.loads(
            (PROCESSED / "forecast_asof_quality_report.json").read_text()
        )
        cls.engineered_quality = json.loads(
            (PROCESSED / "forecast_model_feature_quality_report.json").read_text()
        )

    def test_history_artifact_meets_issue_two_contract(self) -> None:
        history = pd.read_csv(
            PROCESSED / "eirgrid_core_history_30min.csv.gz",
            usecols=[
                "timestamp_utc",
                "dispatch_down_mwh",
                "curtailment_mwh",
                "constraint_mwh",
                "eirgrid_system_available_flag",
                "dispatch_down_available_flag",
            ],
            parse_dates=["timestamp_utc"],
        )
        self.assertEqual(len(history), self.history_quality["rows"])
        self.assertTrue(self.history_quality["meets_24_month_target"])
        self.assertEqual(int(history.duplicated("timestamp_utc").sum()), 0)
        accounting = (
            history["dispatch_down_mwh"]
            - history["curtailment_mwh"]
            - history["constraint_mwh"]
        ).abs()
        self.assertLess(float(accounting.max()), 0.05)
        self.assertEqual(set(history["eirgrid_system_available_flag"]), {1})
        self.assertEqual(set(history["dispatch_down_available_flag"]), {1})

    def test_forecast_artifact_meets_issue_three_contract(self) -> None:
        key = [
            "feature_name",
            "target_timestamp_utc",
            "published_at_utc",
            "revision",
        ]
        vintages = pd.read_csv(
            PROCESSED / "forecast_vintages.csv.gz",
            usecols=[*key, "downloaded_at_utc"],
            parse_dates=[
                "target_timestamp_utc",
                "published_at_utc",
                "downloaded_at_utc",
            ],
        )
        self.assertEqual(len(vintages), self.forecast_quality["rows"])
        self.assertEqual(int(vintages.duplicated(key).sum()), 0)
        self.assertTrue(
            (vintages["published_at_utc"] <= vintages["downloaded_at_utc"]).all()
        )
        required = {
            "wind_forecast_mw_all",
            "solar_forecast_mw_all",
            "demand_forecast_mw_all",
            "interconnector_max_import_mw_roiewic",
            "interconnector_max_import_mw_nimoyle",
            "interconnector_max_import_mw_roigrlk",
        }
        self.assertTrue(required.issubset(set(vintages["feature_name"])))

    def test_asof_artifact_has_no_future_publications(self) -> None:
        asof = pd.read_csv(
            PROCESSED / "forecast_features_asof_30_60.csv",
            parse_dates=["issue_timestamp_utc", "target_timestamp_utc"],
        )
        self.assertEqual(set(asof["forecast_horizon_minutes"]), {30, 60})
        expected_target = asof["issue_timestamp_utc"] + pd.to_timedelta(
            asof["forecast_horizon_minutes"], unit="m"
        )
        self.assertTrue((asof["target_timestamp_utc"] == expected_target).all())
        for column in asof.columns:
            if column.endswith("_published_at_utc"):
                published = pd.to_datetime(asof[column], utc=True)
                self.assertTrue(
                    (published.isna() | (published <= asof["issue_timestamp_utc"])).all(),
                    column,
                )
        self.assertEqual(self.asof_quality["future_publication_violations"], 0)

    def test_manifest_has_complete_provenance(self) -> None:
        manifest = pd.read_csv(PROCESSED / "extended_source_manifest.csv")
        required = {
            "provider",
            "report",
            "year",
            "url",
            "retrieved_at_utc",
            "sha256",
            "licence_note",
            "schema_version",
            "first_interval_utc",
            "last_interval_utc",
            "row_count",
        }
        self.assertTrue(required.issubset(manifest.columns))
        self.assertFalse(manifest[list(required)].isna().any().any())

    def test_engineered_forecast_artifact_meets_issue_four_contract(self) -> None:
        features = pd.read_csv(
            PROCESSED / "forecast_model_features_30_60.csv",
            low_memory=False,
        )
        dictionary = pd.read_csv(
            PROCESSED / "forecast_model_feature_dictionary.csv"
        )
        issue = pd.to_datetime(features["issue_timestamp_utc"], utc=True)
        target = pd.to_datetime(features["target_timestamp_utc"], utc=True)
        expected_target = issue + pd.to_timedelta(
            features["forecast_horizon_minutes"], unit="m"
        )

        self.assertEqual(len(features), self.engineered_quality["rows"])
        self.assertEqual(set(features.columns), set(dictionary["column"]))
        self.assertEqual(
            int(
                features.duplicated(
                    ["issue_timestamp_utc", "forecast_horizon_minutes"]
                ).sum()
            ),
            0,
        )
        self.assertTrue((target == expected_target).all())
        self.assertEqual(self.engineered_quality["future_publication_violations"], 0)
        self.assertEqual(self.engineered_quality["future_observation_violations"], 0)
        self.assertEqual(
            self.engineered_quality["unhandled_missing_feature_columns"], []
        )
        for column in features:
            if column.endswith("_published_at_utc"):
                published = pd.to_datetime(features[column], utc=True)
                self.assertTrue((published.isna() | published.le(issue)).all(), column)
            if column.endswith("_observed_at_utc"):
                observed = pd.to_datetime(features[column], utc=True)
                self.assertTrue((observed.isna() | observed.le(issue)).all(), column)

        # The retained live forecast archive starts after the latest published
        # actuals. The pipeline must flag that state as stale and avoid turning
        # old interconnector flows into apparently current headroom.
        self.assertTrue(features["latest_completed_state_stale_flag"].eq(1).all())
        self.assertTrue(features["interconnector_export_headroom_mw"].isna().all())


if __name__ == "__main__":
    unittest.main()
