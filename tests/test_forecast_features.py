import unittest

import pandas as pd

from gridtoev.eirgrid_history import DataQualityError
from gridtoev.forecast_features import (
    build_forecast_feature_table,
    validate_forecast_feature_table,
)
from gridtoev.forecast_vintages import asof_join_forecasts


FORECAST_FEATURES = {
    "demand_forecast_mw_all": 95.0,
    "solar_forecast_mw_all": 8.0,
    "interconnector_max_export_mw_nimoyle": 100.0,
    "interconnector_max_export_mw_roiewic": 100.0,
    "interconnector_max_export_mw_roigrlk": 100.0,
    "interconnector_max_import_mw_nimoyle": 100.0,
    "interconnector_max_import_mw_roiewic": 100.0,
    "interconnector_max_import_mw_roigrlk": 100.0,
}


def make_history() -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=14, freq="30min", tz="UTC")
    rows = []
    for index, timestamp in enumerate(timestamps):
        rows.append(
            {
                "timestamp_utc": timestamp,
                "eirgrid_all_island_demand_mw": 100.0,
                "eirgrid_all_island_wind_generation_mw": 40.0 + index,
                "eirgrid_all_island_solar_generation_mw": 10.0,
                "eirgrid_ewic_flow_mw": 20.0,
                "eirgrid_moyle_flow_mw": 10.0,
                "eirgrid_greenlink_flow_mw": 5.0,
                "eirgrid_snsp_ratio": 0.5,
                "dispatch_down_mwh": float(index),
                "curtailment_mwh": float(index) * 0.25,
                "constraint_mwh": float(index) * 0.75,
                "dispatch_down_available_flag": 1,
            }
        )
    return pd.DataFrame(rows)


def make_vintages(history: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for index, target in enumerate(history["timestamp_utc"]):
        latest = {**FORECAST_FEATURES, "wind_forecast_mw_all": 38.0 + index}
        previous = {**latest}
        previous["demand_forecast_mw_all"] = 90.0
        previous["wind_forecast_mw_all"] -= 3.0
        previous["solar_forecast_mw_all"] = 6.0
        for revision, published_offset, values in (
            (1, -120, previous),
            (2, -60, latest),
        ):
            published = target + pd.Timedelta(minutes=published_offset)
            for feature_name, value in values.items():
                rows.append(
                    {
                        "feature_name": feature_name,
                        "value": value,
                        "target_timestamp_utc": target,
                        "published_at_utc": published,
                        "downloaded_at_utc": published + pd.Timedelta(minutes=1),
                        "revision": revision,
                        "source_id": f"fixture_revision_{revision}",
                    }
                )
        # This revision was not known at either the 30- or 60-minute issue time.
        for feature_name in latest:
            rows.append(
                {
                    "feature_name": feature_name,
                    "value": 999.0,
                    "target_timestamp_utc": target,
                    "published_at_utc": target,
                    "downloaded_at_utc": target + pd.Timedelta(minutes=1),
                    "revision": 3,
                    "source_id": "fixture_future_revision",
                }
            )
    return pd.DataFrame(rows)


def make_asof(vintages: pd.DataFrame) -> pd.DataFrame:
    rows = pd.DataFrame(
        {
            "issue_timestamp_utc": [
                pd.Timestamp("2026-01-01T05:00:00Z"),
                pd.Timestamp("2026-01-01T04:30:00Z"),
            ],
            "target_timestamp_utc": [
                pd.Timestamp("2026-01-01T05:30:00Z"),
                pd.Timestamp("2026-01-01T05:30:00Z"),
            ],
            "forecast_horizon_minutes": [30, 60],
        }
    )
    return asof_join_forecasts(rows, vintages)


class ForecastFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.history = make_history()
        cls.vintages = make_vintages(cls.history)
        cls.asof = make_asof(cls.vintages)

    def test_builds_state_headroom_revision_and_causal_error_features(self) -> None:
        table, dictionary, quality = build_forecast_feature_table(
            self.asof,
            self.history,
            self.vintages,
        )
        row = table.loc[table["forecast_horizon_minutes"].eq(30)].iloc[0]

        # Target-time forecasts are wind=49 MW, solar=8 MW and demand=95 MW.
        self.assertAlmostEqual(row["forecast_variable_renewable_mw"], 57.0)
        self.assertAlmostEqual(row["forecast_net_load_mw"], 38.0)
        self.assertAlmostEqual(row["forecast_renewable_share_ratio"], 57.0 / 95.0)
        self.assertEqual(row["forecast_renewable_surplus_mw"], 0.0)
        self.assertAlmostEqual(row["interconnector_export_headroom_mw"], 265.0)
        self.assertAlmostEqual(row["interconnector_utilisation_ratio"], 35.0 / 300.0)

        self.assertAlmostEqual(row["demand_forecast_revision_delta_mw"], 5.0)
        self.assertEqual(row["demand_forecast_known_vintage_count"], 2)
        self.assertAlmostEqual(row["wind_forecast_error_mw_latest_completed"], 2.0)
        self.assertAlmostEqual(row["wind_forecast_error_mae_2h"], 2.0)
        self.assertAlmostEqual(row["wind_forecast_error_bias_2h"], 2.0)
        # closed="right" means (issue - 2h, issue], so exactly four completed
        # half-hours are available at this issue time.
        self.assertEqual(row["wind_forecast_error_count_2h"], 4)
        self.assertEqual(
            row["wind_forecast_error_observed_at_utc"],
            row["issue_timestamp_utc"],
        )
        self.assertLessEqual(
            row["wind_forecast_error_observed_at_utc"],
            row["issue_timestamp_utc"],
        )

        self.assertEqual(int(table.duplicated(["issue_timestamp_utc", "forecast_horizon_minutes"]).sum()), 0)
        self.assertEqual(set(table.columns), set(dictionary["column"]))
        self.assertEqual(quality["future_publication_violations"], 0)
        self.assertEqual(quality["future_observation_violations"], 0)
        self.assertEqual(quality["unhandled_missing_feature_columns"], [])

    def test_future_selected_revision_fails_closed(self) -> None:
        bad = self.asof.copy()
        bad["wind_forecast_mw_all_published_at_utc"] = (
            bad["issue_timestamp_utc"] + pd.Timedelta(minutes=1)
        )
        with self.assertRaises(DataQualityError):
            build_forecast_feature_table(bad, self.history, self.vintages)

    def test_future_completed_observation_fails_closed(self) -> None:
        table, _, _ = build_forecast_feature_table(
            self.asof,
            self.history,
            self.vintages,
        )
        table.loc[0, "wind_forecast_error_observed_at_utc"] = (
            table.loc[0, "issue_timestamp_utc"] + pd.Timedelta(minutes=30)
        )
        with self.assertRaises(DataQualityError):
            validate_forecast_feature_table(table)

    def test_duplicate_natural_key_is_rejected(self) -> None:
        duplicate = pd.concat([self.asof, self.asof.iloc[[0]]], ignore_index=True)
        with self.assertRaises(DataQualityError):
            build_forecast_feature_table(duplicate, self.history, self.vintages)


if __name__ == "__main__":
    unittest.main()
