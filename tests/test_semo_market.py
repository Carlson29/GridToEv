import tempfile
import unittest
from pathlib import Path

import pandas as pd

from gridtoev.eirgrid_history import DataQualityError
from gridtoev.semo_market import (
    SUPPORTED_DATASETS,
    build_semo_ablation_record,
    build_semo_feature_matrix,
    parse_semo_market_xml,
    validate_semo_feature_matrix,
)


FIXTURES = Path(__file__).parent / "fixtures" / "semo_market"


def signal(
    feature_name: str,
    value: float,
    published: str,
    *,
    target: str | None = "2026-09-23T22:00:00Z",
    target_end: str | None = "2026-09-23T22:30:00Z",
    resource: str | None = None,
    actual: bool = False,
) -> dict[str, object]:
    return {
        "report_name": "fixture",
        "dataset_name": "fixture",
        "feature_name": feature_name,
        "value": value,
        "unit": "MW",
        "jurisdiction": "ALL",
        "resource_name": resource,
        "resource_type": None,
        "target_timestamp_utc": target,
        "target_end_timestamp_utc": target_end,
        "published_at_utc": published,
        "downloaded_at_utc": "2026-09-23T22:30:00Z",
        "revision": 1,
        "source_id": "fixture",
        "source_url": "https://example.test/fixture.xml",
        "schema_version": "semo-market-v1",
        "actual_outcome_flag": int(actual),
    }


class SemoMarketTests(unittest.TestCase):
    def test_default_price_publication_uses_market_backup_price(self) -> None:
        payload = b"""<?xml version='1.0'?>
<OutboundData DatasetName="PUB_5MinImbalPrc"
 PublishTime="2026-09-23T00:20:01">
  <PUB_5MinImbalPrc StartTime="2026-09-22T23:55:00"
   EndTime="2026-09-23T00:00:00" DefaultPriceUsage="Y"
   MarketBackupPrice="189.67"/>
</OutboundData>"""

        parsed = parse_semo_market_xml(
            payload,
            source_id="default_price",
            source_url="https://example.test/default-price.xml",
            downloaded_at="2026-09-23T00:21:00Z",
        )

        self.assertEqual(len(parsed), 1)
        self.assertEqual(
            parsed.iloc[0]["feature_name"],
            "imbalance_price_5min_eur_per_mwh",
        )
        self.assertEqual(parsed.iloc[0]["value"], 189.67)

    def test_all_observed_report_schemas_have_parser_fixtures(self) -> None:
        frames = []
        for fixture in sorted(FIXTURES.glob("*.xml")):
            frames.append(
                parse_semo_market_xml(
                    fixture,
                    source_id=fixture.stem,
                    source_url=f"https://example.test/{fixture.name}",
                    downloaded_at="2026-09-23T22:30:00Z",
                )
            )
        parsed = pd.concat(frames, ignore_index=True)

        self.assertEqual(set(parsed["dataset_name"]), set(SUPPORTED_DATASETS))
        self.assertEqual(set(parsed["schema_version"]), {"semo-market-v1"})
        self.assertTrue(parsed["value"].notna().all())
        self.assertTrue(parsed["unit"].notna().all())
        self.assertTrue(parsed["jurisdiction"].notna().all())

        schedule = parsed.loc[
            parsed["dataset_name"].eq("PUB_RTCOperationalSchedule")
        ].iloc[0]
        self.assertEqual(
            schedule["published_at_utc"].isoformat(),
            "2026-09-23T20:05:00+00:00",
        )
        price = parsed.loc[parsed["dataset_name"].eq("PUB_30MinAvgImbalPrc")]
        self.assertTrue(price["actual_outcome_flag"].eq(1).all())
        self.assertTrue(
            (price["published_at_utc"] >= price["target_end_timestamp_utc"]).all()
        )

    def test_feature_builder_is_causal_and_engineers_operational_signals(self) -> None:
        rows = [
            signal(
                "rtc_scheduled_quantity_mw",
                4000,
                "2026-09-23T21:00:00Z",
                resource="GEN_1",
            ),
            signal(
                "rtc_scheduled_quantity_mw",
                100,
                "2026-09-23T21:00:00Z",
                resource="I_NIMOYLE",
            ),
            signal("demand_forecast_mw_all", 4600, "2026-09-23T20:00:00Z"),
            signal("forecast_calculated_imbalance_mw", 100, "2026-09-23T21:00:00Z"),
            signal("aggregated_pn_start_mw", 3900, "2026-09-23T20:30:00Z"),
            signal("aggregated_pn_end_mw", 3950, "2026-09-23T20:30:00Z"),
            signal(
                "interconnector_max_export_mw_nimoyle",
                500,
                "2026-09-23T06:00:00Z",
                target_end="2026-09-23T23:00:00Z",
            ),
            signal(
                "interconnector_max_export_mw_roiewic",
                500,
                "2026-09-23T06:00:00Z",
                target_end="2026-09-23T23:00:00Z",
            ),
            signal(
                "interconnector_max_export_mw_roigrlk",
                500,
                "2026-09-23T06:00:00Z",
                target_end="2026-09-23T23:00:00Z",
            ),
            signal(
                "imbalance_price_30min_eur_per_mwh",
                50,
                "2026-09-23T21:05:00Z",
                target="2026-09-23T20:30:00Z",
                target_end="2026-09-23T21:00:00Z",
                actual=True,
            ),
            signal(
                "imbalance_price_30min_eur_per_mwh",
                999,
                "2026-09-23T21:31:00Z",
                target="2026-09-23T21:00:00Z",
                target_end="2026-09-23T21:30:00Z",
                actual=True,
            ),
        ]
        modelling = pd.DataFrame(
            {
                "issue_timestamp_utc": ["2026-09-23T21:30:00Z"],
                "target_timestamp_utc": ["2026-09-23T22:00:00Z"],
                "forecast_horizon_minutes": [30],
            }
        )
        history = pd.DataFrame(
            {
                "timestamp_utc": ["2026-09-23T21:00:00Z"],
                "eirgrid_all_island_generation_mw": [3800.0],
                "eirgrid_moyle_flow_mw": [10.0],
                "eirgrid_ewic_flow_mw": [20.0],
                "eirgrid_greenlink_flow_mw": [5.0],
            }
        )

        features, dictionary, quality = build_semo_feature_matrix(
            modelling,
            pd.DataFrame(rows),
            history,
        )
        result = features.iloc[0]

        self.assertEqual(result["rtc_scheduled_generation_mw"], 4000)
        self.assertEqual(result["rtc_interconnector_schedule_mw"], 100)
        self.assertEqual(result["scheduled_minus_actual_generation_mw"], 200)
        self.assertEqual(result["downward_margin_proxy_mw"], 0)
        self.assertEqual(result["lagged_imbalance_price_eur_per_mwh"], 50)
        self.assertEqual(result["interconnector_export_headroom_mw"], 1465)
        self.assertEqual(set(features.columns), set(dictionary["column"]))
        self.assertEqual(quality["future_publication_violations"], 0)
        self.assertEqual(quality["future_observation_violations"], 0)

    def test_future_publication_is_rejected_by_validator(self) -> None:
        table = pd.DataFrame(
            {
                "issue_timestamp_utc": ["2026-09-23T21:30:00Z"],
                "target_timestamp_utc": ["2026-09-23T22:00:00Z"],
                "forecast_horizon_minutes": [30],
                "signal_published_at_utc": ["2026-09-23T21:31:00Z"],
            }
        )
        with self.assertRaises(DataQualityError):
            validate_semo_feature_matrix(table)

    def test_ablation_is_recorded_as_rejected_without_temporal_overlap(self) -> None:
        baseline = pd.DataFrame(
            {
                "issue_timestamp_utc": pd.to_datetime(
                    ["2026-01-01T00:00:00Z", "2026-01-01T00:30:00Z"], utc=True
                )
            }
        )
        market = pd.DataFrame(
            {
                "issue_timestamp_utc": pd.to_datetime(
                    ["2026-09-23T21:00:00Z"], utc=True
                ),
                "forecast_horizon_minutes": [30],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            signal_path = Path(directory) / "signals.csv"
            signal_path.write_text("feature,value\nprice,50\n", encoding="utf-8")
            report = build_semo_ablation_record(baseline, market, signal_path)

        self.assertEqual(report["experiment_name"], "semo_market_operational_signals")
        self.assertEqual(report["overlap_rows"], 0)
        self.assertEqual(report["status"], "rejected_no_temporal_overlap")
        self.assertEqual(report["production_decision"], "exclude")


if __name__ == "__main__":
    unittest.main()
