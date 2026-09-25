import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from gridtoev.eirgrid_history import DataQualityError
from gridtoev.regional_weather import (
    build_weather_ablation_report,
    build_regional_weather_features,
    parse_open_meteo_analysis,
    parse_open_meteo_single_run,
    validate_region_weights,
    validate_weather_feature_matrix,
)


REGIONS = pd.DataFrame(
    {
        "region": ["west", "east"],
        "latitude": [53.3, 53.3],
        "longitude": [-9.0, -6.2],
        "wind_capacity_mw": [300.0, 100.0],
        "solar_capacity_mw": [100.0, 300.0],
    }
)


class RegionalWeatherParserTests(unittest.TestCase):
    def test_single_run_parser_preserves_run_valid_and_conservative_publish_times(self) -> None:
        payload = [
            {
                "latitude": 53.3,
                "longitude": -9.0,
                "hourly_units": {
                    "wind_speed_100m": "m/s",
                    "shortwave_radiation": "W/m²",
                },
                "hourly": {
                    "time": ["2026-01-01T00:00", "2026-01-01T01:00"],
                    "wind_speed_100m": [10.0, 12.0],
                    "wind_direction_100m": [90.0, 100.0],
                    "wind_gusts_10m": [15.0, 17.0],
                    "temperature_2m": [5.0, 6.0],
                    "pressure_msl": [1000.0, 999.0],
                    "cloud_cover": [80.0, 70.0],
                    "shortwave_radiation": [0.0, 5.0],
                },
            },
            {
                "latitude": 53.3,
                "longitude": -6.2,
                "hourly_units": {},
                "hourly": {
                    "time": ["2026-01-01T00:00", "2026-01-01T01:00"],
                    "wind_speed_100m": [4.0, 5.0],
                    "wind_direction_100m": [180.0, 190.0],
                    "wind_gusts_10m": [7.0, 8.0],
                    "temperature_2m": [7.0, 8.0],
                    "pressure_msl": [1002.0, 1001.0],
                    "cloud_cover": [60.0, 50.0],
                    "shortwave_radiation": [0.0, 10.0],
                },
            },
        ]

        parsed = parse_open_meteo_single_run(
            payload,
            REGIONS,
            run_initialized_at="2026-01-01T00:00:00Z",
            downloaded_at="2026-02-01T00:00:00Z",
            source_url="https://example.test/single-run",
        )

        self.assertEqual(len(parsed), 4)
        self.assertEqual(set(parsed["region"]), {"west", "east"})
        self.assertTrue(
            (
                parsed["published_at_utc"]
                == pd.Timestamp("2026-01-01T06:00:00Z")
            ).all()
        )
        self.assertTrue(
            (parsed["run_initialized_at_utc"] < parsed["published_at_utc"]).all()
        )
        self.assertEqual(parsed.loc[0, "wind_speed_100m_mps"], 10.0)

    def test_analysis_parser_assigns_an_explicit_availability_lag(self) -> None:
        payload = {
            "latitude": 53.3,
            "longitude": -9.0,
            "hourly": {
                "time": ["2026-01-01T00:00"],
                "wind_speed_100m": [11.0],
                "wind_direction_100m": [90.0],
                "wind_gusts_10m": [14.0],
                "temperature_2m": [5.0],
                "pressure_msl": [1000.0],
                "cloud_cover": [80.0],
                "shortwave_radiation": [0.0],
            },
        }
        parsed = parse_open_meteo_analysis(
            payload,
            REGIONS.iloc[[0]],
            downloaded_at="2026-02-01T00:00:00Z",
            source_url="https://example.test/analysis",
        )
        self.assertEqual(
            parsed.loc[0, "published_at_utc"],
            pd.Timestamp("2026-01-01T06:00:00Z"),
        )
        self.assertEqual(parsed.loc[0, "source_role"], "analysis_proxy")

    def test_region_weights_reject_duplicates_negative_capacity_and_zero_totals(self) -> None:
        duplicate = pd.concat([REGIONS, REGIONS.iloc[[0]]], ignore_index=True)
        with self.assertRaises(DataQualityError):
            validate_region_weights(duplicate)
        negative = REGIONS.copy()
        negative.loc[0, "wind_capacity_mw"] = -1
        with self.assertRaises(DataQualityError):
            validate_region_weights(negative)


class RegionalWeatherFeatureTests(unittest.TestCase):
    def test_feature_builder_uses_only_known_vintages_and_flags_missing_regions(self) -> None:
        modelling = pd.DataFrame(
            {
                "issue_timestamp_utc": [
                    "2026-01-01T10:00:00Z",
                    "2026-01-01T13:00:00Z",
                ],
                "target_timestamp_utc": [
                    "2026-01-01T10:30:00Z",
                    "2026-01-01T13:30:00Z",
                ],
                "forecast_horizon_minutes": [30, 30],
            }
        )
        rows: list[dict[str, object]] = []
        for region, speed, capacity_time_end in (
            ("west", 10.0, "2026-01-01T15:00:00Z"),
            ("east", 2.0, "2026-01-01T12:00:00Z"),
        ):
            for valid in pd.date_range(
                "2026-01-01T07:00:00Z", capacity_time_end, freq="1h"
            ):
                rows.append(
                    weather_row(
                        region=region,
                        run="2026-01-01T00:00:00Z",
                        published="2026-01-01T06:00:00Z",
                        valid=valid,
                        speed=speed,
                    )
                )
        for valid in pd.date_range(
            "2026-01-01T13:00:00Z", "2026-01-01T15:00:00Z", freq="1h"
        ):
            rows.append(
                weather_row(
                    region="west",
                    run="2026-01-01T06:00:00Z",
                    published="2026-01-01T12:00:00Z",
                    valid=valid,
                    speed=20.0,
                )
            )
        # This revision exists in the data but is not yet public at either issue.
        rows.append(
            weather_row(
                region="west",
                run="2026-01-01T12:00:00Z",
                published="2026-01-01T18:00:00Z",
                valid=pd.Timestamp("2026-01-01T13:00:00Z"),
                speed=99.0,
            )
        )
        forecasts = pd.DataFrame(rows)

        features, dictionary, quality = build_regional_weather_features(
            modelling,
            forecasts,
            pd.DataFrame(),
            REGIONS,
        )

        first = features.iloc[0]
        second = features.iloc[1]
        self.assertAlmostEqual(first["weather_wind_weighted_speed_100m_mps"], 8.0)
        self.assertAlmostEqual(first["weather_wind_region_coverage_ratio"], 1.0)
        self.assertEqual(first["weather_region_east_missing_flag"], 0)
        self.assertAlmostEqual(second["weather_wind_weighted_speed_100m_mps"], 20.0)
        self.assertAlmostEqual(second["weather_wind_region_coverage_ratio"], 0.75)
        self.assertEqual(second["weather_region_east_missing_flag"], 1)
        self.assertLessEqual(
            second["weather_forecast_published_at_utc"],
            second["issue_timestamp_utc"],
        )
        self.assertEqual(set(features.columns), set(dictionary["column"]))
        self.assertEqual(quality["future_publication_violations"], 0)

    def test_validator_rejects_future_weather_publication(self) -> None:
        table = pd.DataFrame(
            {
                "issue_timestamp_utc": ["2026-01-01T10:00:00Z"],
                "target_timestamp_utc": ["2026-01-01T10:30:00Z"],
                "forecast_horizon_minutes": [30],
                "weather_forecast_published_at_utc": ["2026-01-01T11:00:00Z"],
            }
        )
        with self.assertRaises(DataQualityError):
            validate_weather_feature_matrix(table)

    def test_ablation_keeps_final_test_sealed_and_accepts_strong_signal(self) -> None:
        issue = pd.date_range("2026-01-01", periods=200, freq="30min", tz="UTC")
        signal = np.tile([0.0, 100.0], 100)
        model_data = pd.DataFrame(
            {
                "issue_timestamp_utc": issue,
                "target_timestamp_utc": issue + pd.Timedelta(minutes=30),
                "forecast_horizon_minutes": 30,
                "dispatch_down_mwh_latest_observed": 0.0,
                "baseline_noise": np.sin(np.arange(200)),
                "dispatch_down_mwh": signal,
            }
        )
        weather = model_data[
            [
                "issue_timestamp_utc",
                "target_timestamp_utc",
                "forecast_horizon_minutes",
            ]
        ].copy()
        weather["weather_wind_weighted_speed_100m_mps"] = signal
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "weather.csv"
            weather.to_csv(artifact, index=False)
            report = build_weather_ablation_report(model_data, weather, artifact)

        self.assertEqual(report["experiment_name"], "regional_weather_signals")
        self.assertEqual(report["development_fold_count"], 4)
        self.assertIn("selected_variant", report)
        self.assertIn(report["selected_variant"], report["variants"])
        self.assertEqual(report["production_decision"], "include")
        self.assertGreater(report["metrics"]["mean_mae_improvement_fraction"], 0.5)
        self.assertFalse(report["final_test_accessed"])


def weather_row(
    *,
    region: str,
    run: str,
    published: str,
    valid: pd.Timestamp,
    speed: float,
) -> dict[str, object]:
    return {
        "region": region,
        "latitude": 53.3,
        "longitude": -9.0,
        "model": "ecmwf_ifs",
        "run_initialized_at_utc": run,
        "published_at_utc": published,
        "valid_timestamp_utc": valid,
        "lead_hours": (valid - pd.Timestamp(run)).total_seconds() / 3600,
        "wind_speed_100m_mps": speed,
        "wind_direction_100m_deg": 90.0,
        "wind_gusts_10m_mps": speed + 3.0,
        "temperature_2m_c": 5.0,
        "pressure_msl_hpa": 1000.0,
        "cloud_cover_pct": 50.0,
        "shortwave_radiation_wm2": 100.0,
        "downloaded_at_utc": "2026-02-01T00:00:00Z",
        "source_url": "https://example.test/run",
        "schema_version": "regional-weather-v1",
    }


if __name__ == "__main__":
    unittest.main()
