import unittest

import pandas as pd

from gridtoev.forecast_vintages import (
    asof_join_forecasts,
    deduplicate_vintages,
    parse_semo_forecast_xml,
    parse_smartgrid_snapshot,
)


WIND_XML = b"""<?xml version='1.0'?>
<OutboundData DatasetName="PUB_15MinAggWindFcst" PublishTime="2024-01-01T09:40:00">
  <PUB_15MinAggWindFcst ROW="1" StartTime="2024-01-01T10:00:00"
      EndTime="2024-01-01T10:15:00" Forecast="1234.5" />
</OutboundData>
"""

LOAD_XML = b"""<?xml version='1.0'?>
<OutboundData DatasetName="PUB_DailyLoadFcst" PublishTime="2024-01-01T09:35:00">
  <PUB_DailyLoadFcst ROW="1">
    <StartTime>2024-01-01T10:00:00</StartTime>
    <EndTime>2024-01-01T10:30:00</EndTime>
    <LoadForecastROI>4000</LoadForecastROI>
    <LoadForecastNI>800</LoadForecastNI>
    <AggregatedForecast>4800</AggregatedForecast>
  </PUB_DailyLoadFcst>
</OutboundData>
"""


class ForecastVintageTests(unittest.TestCase):
    def test_smartgrid_wide_solar_snapshot_uses_retrieval_as_publication(self) -> None:
        snapshot = {
            "Rows": [
                {
                    "Effective_Date": "23-09-2026 15:00:00",
                    "MW_ACTUAL": "500.0",
                    "MW_FORECAST": "525.5",
                    "Region": "ALL",
                }
            ]
        }

        result = parse_smartgrid_snapshot(
            snapshot,
            area="solar",
            source_id="solar_snapshot",
            source_url="https://www.vargen.smartgriddashboard.com/api/export/example",
            downloaded_at="2026-09-23T14:30:00Z",
        )

        self.assertEqual(result.loc[0, "feature_name"], "solar_forecast_mw_all")
        self.assertEqual(float(result.loc[0, "value"]), 525.5)
        self.assertEqual(
            result.loc[0, "published_at_utc"], result.loc[0, "downloaded_at_utc"]
        )
        self.assertEqual(
            result.loc[0, "target_timestamp_utc"].isoformat(),
            "2026-09-23T15:00:00+00:00",
        )

    def test_semo_xml_keeps_target_publication_and_download_times(self) -> None:
        wind = parse_semo_forecast_xml(
            WIND_XML,
            source_id="semo_wind",
            source_url="https://reports.sem-o.com/documents/wind.xml",
            downloaded_at="2024-01-01T09:45:00Z",
        )
        load = parse_semo_forecast_xml(
            LOAD_XML,
            source_id="semo_load",
            source_url="https://reports.sem-o.com/documents/load.xml",
            downloaded_at="2024-01-01T09:45:00Z",
        )

        self.assertEqual(wind.loc[0, "feature_name"], "wind_forecast_mw_all")
        self.assertEqual(float(wind.loc[0, "value"]), 1234.5)
        self.assertEqual(load["feature_name"].tolist(), [
            "demand_forecast_mw_roi",
            "demand_forecast_mw_ni",
            "demand_forecast_mw_all",
        ])
        self.assertTrue((wind["published_at_utc"] < wind["downloaded_at_utc"]).all())
        self.assertEqual(wind.loc[0, "target_time_original"], "2024-01-01T10:00:00")

    def test_duplicate_revision_keeps_latest_download(self) -> None:
        rows = pd.DataFrame(
            {
                "feature_name": ["wind_forecast_mw_all"] * 2,
                "target_timestamp_utc": pd.to_datetime(
                    ["2024-01-01T10:00Z", "2024-01-01T10:00Z"], utc=True
                ),
                "published_at_utc": pd.to_datetime(
                    ["2024-01-01T09:40Z", "2024-01-01T09:40Z"], utc=True
                ),
                "revision": [1, 1],
                "downloaded_at_utc": pd.to_datetime(
                    ["2024-01-01T09:41Z", "2024-01-01T09:42Z"], utc=True
                ),
                "value": [1000.0, 1001.0],
            }
        )

        result = deduplicate_vintages(rows)

        self.assertEqual(len(result), 1)
        self.assertEqual(float(result.loc[0, "value"]), 1001.0)

    def test_asof_join_excludes_late_files_and_reproduces_30_60_minute_values(self) -> None:
        issues = pd.DataFrame(
            {
                "issue_timestamp_utc": pd.to_datetime(
                    ["2024-01-01T09:30Z", "2024-01-01T09:30Z"], utc=True
                ),
                "target_timestamp_utc": pd.to_datetime(
                    ["2024-01-01T10:00Z", "2024-01-01T10:30Z"], utc=True
                ),
                "forecast_horizon_minutes": [30, 60],
            }
        )
        vintages = pd.DataFrame(
            {
                "feature_name": ["wind_forecast_mw_all"] * 4,
                "target_timestamp_utc": pd.to_datetime(
                    [
                        "2024-01-01T10:00Z",
                        "2024-01-01T10:00Z",
                        "2024-01-01T10:30Z",
                        "2024-01-01T10:30Z",
                    ],
                    utc=True,
                ),
                "published_at_utc": pd.to_datetime(
                    [
                        "2024-01-01T09:00Z",
                        "2024-01-01T09:31Z",
                        "2024-01-01T09:15Z",
                        "2024-01-01T09:45Z",
                    ],
                    utc=True,
                ),
                "downloaded_at_utc": pd.to_datetime(
                    [
                        "2024-01-01T09:01Z",
                        "2024-01-01T09:32Z",
                        "2024-01-01T09:16Z",
                        "2024-01-01T09:46Z",
                    ],
                    utc=True,
                ),
                "revision": [1, 2, 1, 2],
                "value": [1000.0, 9999.0, 1200.0, 9999.0],
            }
        )

        result = asof_join_forecasts(
            issues,
            vintages,
            feature_names=["wind_forecast_mw_all"],
        )

        self.assertEqual(result["wind_forecast_mw_all"].tolist(), [1000.0, 1200.0])
        published = result["wind_forecast_mw_all_published_at_utc"]
        self.assertTrue((published <= result["issue_timestamp_utc"]).all())
        self.assertEqual(result["wind_forecast_mw_all_available_flag"].tolist(), [1, 1])

        unavailable = asof_join_forecasts(
            issues,
            vintages,
            feature_names=["solar_forecast_mw_all"],
        )
        self.assertEqual(unavailable["solar_forecast_mw_all_available_flag"].tolist(), [0, 0])
        self.assertTrue(unavailable["solar_forecast_mw_all"].isna().all())


if __name__ == "__main__":
    unittest.main()
