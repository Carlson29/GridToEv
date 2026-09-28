import json
import unittest
from pathlib import Path

import pandas as pd

from gridtoev.eirgrid_history import DataQualityError
from gridtoev.source_curtailment import (
    build_daily_source_labels,
    build_source_label_table,
    normalise_source_dispatch_frame,
    reconcile_daily_with_v2,
    reconcile_with_core,
)


ROOT = Path(__file__).resolve().parents[1]


def _raw(rows: list[tuple[str, str, str, int, float]]) -> pd.DataFrame:
    """rows: (UT_TYPE, JURISDICTION, local timestamp, GMT offset, curtailment)."""
    return pd.DataFrame({
        "UT_TYPE": [row[0] for row in rows],
        "JURISDICTION": [row[1] for row in rows],
        "HH_TIMESTAMP": [row[2] for row in rows],
        "GMT_OFFSET": [row[3] for row in rows],
        "Sum of DD_MWH": [row[4] + 1.0 for row in rows],
        "Sum of CURTAILMENTS_MWH": [row[4] for row in rows],
        "Sum of CONSTRAINTS_MWH": [1.0 for _ in rows],
    })


def _day(day: str, wind: float, solar: float | None) -> pd.DataFrame:
    stamps = pd.date_range(day, periods=48, freq="30min").strftime("%Y-%m-%d %H:%M")
    rows = [("Wind", "IE", stamp, 0, wind) for stamp in stamps]
    if solar is not None:
        rows += [("Solar", "IE", stamp, 0, solar) for stamp in stamps]
    rows += [("Wind", "NI", stamp, 0, 99.0) for stamp in stamps]
    return _raw(rows)


class SourceLabelTests(unittest.TestCase):
    def test_ut_type_mapping_filters_ie_and_keeps_both_sources(self) -> None:
        frame = normalise_source_dispatch_frame(_day("2024-06-01", 2.0, 0.5), source_id="dd_2024")
        self.assertEqual(set(frame["jurisdiction"]), {"IE"})
        self.assertEqual(frame.groupby("technology_type").size().to_dict(), {"Solar": 48, "Wind": 48})

    def test_unknown_technology_is_rejected(self) -> None:
        raw = _raw([("Hydro", "IE", "2024-06-01 00:00", 0, 1.0)])
        with self.assertRaisesRegex(DataQualityError, "unaudited IE technology"):
            normalise_source_dispatch_frame(raw, source_id="dd")

    def test_duplicate_raw_key_is_rejected_not_summed(self) -> None:
        raw = _raw([("Wind", "IE", "2024-06-01 00:00", 0, 1.0), ("Wind", "IE", "2024-06-01 00:00", 0, 2.0)])
        with self.assertRaisesRegex(DataQualityError, "duplicate raw source-label keys"):
            normalise_source_dispatch_frame(raw, source_id="dd")

    def test_negative_label_is_rejected(self) -> None:
        raw = _raw([("Wind", "IE", "2024-06-01 00:00", 0, -1.0)])
        raw["Sum of DD_MWH"] = 0.0  # keep accounting consistent so the sign check fires
        with self.assertRaisesRegex(DataQualityError, "negative source labels"):
            normalise_source_dispatch_frame(raw, source_id="dd")

    def test_dst_fallback_hour_maps_to_unique_utc_keys(self) -> None:
        raw = _raw([
            ("Wind", "IE", "2024-10-27 01:00", 1, 1.0), ("Wind", "IE", "2024-10-27 01:00", 0, 2.0),
            ("Solar", "IE", "2024-10-27 01:00", 1, 0.0), ("Solar", "IE", "2024-10-27 01:00", 0, 0.0),
        ])
        labels = build_source_label_table([normalise_source_dispatch_frame(raw, source_id="dd")])
        self.assertEqual(
            labels["timestamp_utc"].dt.strftime("%H:%M").tolist(), ["00:00", "00:30", "01:00"],
        )
        self.assertEqual(labels["wind_curtailment_mwh"].tolist()[0::2], [1.0, 2.0])
        # The gap half-hour exists on the grid but is not reported by either source.
        self.assertEqual(labels["source_labels_complete_flag"].tolist(), [1, 0, 1])

    def test_missing_solar_is_unknown_but_reported_zero_is_zero(self) -> None:
        frames = [
            normalise_source_dispatch_frame(_day("2023-03-31", 3.0, None), source_id="dd_a"),
            normalise_source_dispatch_frame(_day("2023-04-01", 3.0, 0.0), source_id="dd_b"),
        ]
        labels = build_source_label_table(frames)
        wind_only = labels.loc[labels["timestamp_utc"].lt("2023-04-01T00:00:00Z")]
        both = labels.loc[labels["timestamp_utc"].ge("2023-04-01T00:00:00Z")]
        self.assertTrue(wind_only["solar_curtailment_mwh"].isna().all())
        self.assertTrue(wind_only["source_curtailment_total_mwh"].isna().all())
        self.assertTrue(both["solar_curtailment_mwh"].eq(0).all())
        daily = build_daily_source_labels(labels)
        # The partial-solar day is excluded rather than treated as a zero-solar day.
        self.assertEqual(daily["target_date_utc"].dt.strftime("%Y-%m-%d").tolist(), ["2023-04-01"])
        self.assertAlmostEqual(float(daily.loc[0, "source_curtailment_total_mwh"]), 144.0)
        self.assertAlmostEqual(float(daily.loc[0, "wind_share"]), 1.0)

    def test_daily_rollup_requires_all_48_half_hours(self) -> None:
        raw = _day("2024-06-01", 1.0, 1.0).iloc[2:]  # drop the first two Wind half-hours
        labels = build_source_label_table([normalise_source_dispatch_frame(raw, source_id="dd")])
        self.assertTrue(build_daily_source_labels(labels).empty)

    def test_zero_curtailment_day_has_no_share(self) -> None:
        labels = build_source_label_table([normalise_source_dispatch_frame(_day("2024-06-01", 0.0, 0.0), source_id="dd")])
        daily = build_daily_source_labels(labels)
        self.assertEqual(float(daily.loc[0, "source_curtailment_total_mwh"]), 0.0)
        self.assertTrue(pd.isna(daily.loc[0, "wind_share"]))

    def test_reconciliation_rejects_mismatched_core_total(self) -> None:
        labels = build_source_label_table([normalise_source_dispatch_frame(_day("2024-06-01", 2.0, 1.0), source_id="dd")])
        core = labels[["timestamp_utc"]].assign(curtailment_mwh=3.0)
        self.assertEqual(reconcile_with_core(labels, core)["complete_rows_compared"], 48)
        core.loc[5, "curtailment_mwh"] = 3.2
        with self.assertRaisesRegex(DataQualityError, "do not reconcile"):
            reconcile_with_core(labels, core)
        daily = build_daily_source_labels(labels)
        v2 = pd.DataFrame({"issue_timestamp_utc": ["2024-06-01T00:00:00Z"], "curtailment_mwh": [144.0]})
        self.assertEqual(reconcile_daily_with_v2(daily, v2)["matched_days"], 1)
        v2["curtailment_mwh"] = 150.0
        with self.assertRaisesRegex(DataQualityError, "V2 labels"):
            reconcile_daily_with_v2(daily, v2)


class SourceLabelArtifactTests(unittest.TestCase):
    """The committed label artifacts must match their saved audit."""

    def setUp(self) -> None:
        self.report = json.loads(
            (ROOT / "data/processed/eirgrid_source_curtailment_quality_report.json").read_text(encoding="utf-8")
        )

    def test_committed_labels_reconcile_and_match_report(self) -> None:
        labels = pd.read_csv(ROOT / "data/processed/eirgrid_source_curtailment_30min.csv.gz", parse_dates=["timestamp_utc"])
        daily = pd.read_csv(ROOT / "data/processed/eirgrid_source_curtailment_daily.csv")
        self.assertEqual(len(labels), self.report["rows"])
        self.assertEqual(int(labels["source_labels_complete_flag"].sum()), self.report["complete_half_hours"])
        self.assertFalse(labels["timestamp_utc"].duplicated().any())
        self.assertEqual(len(daily), self.report["daily"]["complete_days"])
        self.assertTrue((daily[["wind_curtailment_mwh", "solar_curtailment_mwh"]] >= 0).all().all())
        self.assertLess(self.report["core_reconciliation"]["max_abs_difference_mwh"], 0.05)
        self.assertLess(self.report["v2_daily_reconciliation"]["max_abs_difference_mwh"], 0.05)
        self.assertEqual(self.report["v2_daily_reconciliation"]["v2_days_without_complete_source_labels"], [])

    def test_solar_era_starts_april_2023_and_is_not_a_zero_regime(self) -> None:
        self.assertEqual(self.report["by_year"]["2022"]["solar"]["reported_half_hours"], 0)
        self.assertIsNone(self.report["by_year"]["2022"]["solar_share_of_complete_curtailment"])
        self.assertEqual(self.report["by_year"]["2023"]["solar"]["first_reported_utc"], "2023-03-31T23:00:00+00:00")
        self.assertEqual(self.report["daily"]["first_day_utc"], "2023-04-01")


if __name__ == "__main__":
    unittest.main()
