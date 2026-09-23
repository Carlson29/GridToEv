import unittest

import pandas as pd

from gridtoev.eirgrid_history import (
    DataQualityError,
    build_core_history,
    normalise_dispatch_frame,
    normalise_system_frame,
)


class EirGridHistoryTests(unittest.TestCase):
    def test_gmt_offset_disambiguates_dst_fallback(self) -> None:
        raw = pd.DataFrame(
            {
                "DateTime": ["2024-10-27 01:00", "2024-10-27 01:00"],
                "GMT Offset": [1, 0],
                "IE Demand": [4000.0, 4100.0],
                "IE Wind Generation": [1000.0, 1100.0],
            }
        )

        result = normalise_system_frame(raw, source_id="system_2024")

        self.assertEqual(
            result["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%MZ").tolist(),
            ["2024-10-27T00:00Z", "2024-10-27T01:00Z"],
        )
        self.assertEqual(result["timestamp_local_original"].nunique(), 1)
        self.assertEqual(int(result.duplicated("timestamp_utc").sum()), 0)

    def test_duplicate_system_timestamp_is_rejected(self) -> None:
        raw = pd.DataFrame(
            {
                "DateTime": ["2024-01-01 00:00", "2024-01-01 00:00"],
                "GMT Offset": [0, 0],
                "IE Demand": [4000.0, 4001.0],
            }
        )

        with self.assertRaisesRegex(DataQualityError, "duplicate UTC timestamps"):
            normalise_system_frame(raw, source_id="system_2024")

    def test_dispatch_accounting_and_core_history_flags(self) -> None:
        system_raw = pd.DataFrame(
            {
                "DateTime": pd.date_range("2024-01-01", periods=4, freq="15min"),
                "GMT Offset": [0, 0, 0, 0],
                "IE Demand": [4000.0, 4020.0, 4040.0, 4060.0],
                "IE Wind Generation": [1000.0, 1020.0, 1040.0, 1060.0],
            }
        )
        dispatch_raw = pd.DataFrame(
            {
                "UT_TYPE": ["Wind"],
                "JURISDICTION": ["IE"],
                "HH_TIMESTAMP": ["2024-01-01 00:00"],
                "GMT_OFFSET": [0],
                "Sum of DD_MWH": [12.0],
                "Sum of CURTAILMENTS_MWH": [5.0],
                "Sum of CONSTRAINTS_MWH": [7.0],
                "Sum of OTHER_MWH": [0.0],
            }
        )
        system = normalise_system_frame(system_raw, source_id="system_2024")
        dispatch = normalise_dispatch_frame(dispatch_raw, source_id="dispatch_2024")

        core, quality = build_core_history([system], [dispatch])

        self.assertEqual(len(core), 2)
        self.assertEqual(core["eirgrid_system_available_flag"].tolist(), [1, 1])
        self.assertEqual(
            core["eirgrid_ie_demand_mw_available_flag"].tolist(), [1, 1]
        )
        self.assertEqual(core["dispatch_down_available_flag"].tolist(), [1, 0])
        self.assertEqual(float(core.loc[0, "dispatch_down_mwh"]), 12.0)
        self.assertTrue(pd.isna(core.loc[1, "dispatch_down_mwh"]))
        self.assertEqual(quality["duplicate_natural_keys"], 0)
        self.assertEqual(quality["max_dispatch_accounting_difference_mwh"], 0.0)


if __name__ == "__main__":
    unittest.main()
