import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gridtoev.history_model import SIGNALS, build_causal_history_features, model_feature_columns


def history_fixture(rows: int = 110) -> pd.DataFrame:
    times = pd.date_range("2025-01-01", periods=rows, freq="30min", tz="UTC")
    frame = pd.DataFrame({"timestamp_utc": times})
    for signal in SIGNALS:
        frame[signal] = np.arange(rows, dtype=float)
    return frame


class HistoryModelTests(unittest.TestCase):
    def test_delayed_snapshot_and_exact_future_label(self) -> None:
        frame = build_causal_history_features(history_fixture(), observation_delay_hours=24)
        row = frame.loc[
            frame["issue_timestamp_utc"].eq(pd.Timestamp("2025-01-02T00:00:00Z"))
            & frame["forecast_horizon_minutes"].eq(30)
        ].iloc[0]
        self.assertEqual(row["feature_timestamp_utc"], pd.Timestamp("2025-01-01T00:00:00Z"))
        self.assertEqual(row["dispatch_down_mwh_asof"], 0)
        self.assertEqual(row["dispatch_down_mwh"], 49)
        self.assertEqual(row["feature_age_hours"], 24)
        self.assertNotIn("feature_timestamp_utc", model_feature_columns(frame))
        self.assertNotIn("dispatch_down_mwh", model_feature_columns(frame))

    def test_future_mutation_does_not_change_earlier_features(self) -> None:
        original = history_fixture()
        changed = original.copy()
        changed.loc[changed.index >= 60, "dispatch_down_mwh"] = 9999
        before = build_causal_history_features(original, observation_delay_hours=24)
        after = build_causal_history_features(changed, observation_delay_hours=24)
        issue = pd.Timestamp("2025-01-02T06:00:00Z")
        columns = model_feature_columns(before)
        pd.testing.assert_frame_equal(
            before.loc[before["issue_timestamp_utc"].eq(issue), columns].reset_index(drop=True),
            after.loc[after["issue_timestamp_utc"].eq(issue), columns].reset_index(drop=True),
        )

    def test_rejects_unqualified_delay_and_duplicate_source_times(self) -> None:
        with self.assertRaises(ValueError):
            build_causal_history_features(history_fixture(), observation_delay_hours=0)
        duplicated = history_fixture()
        duplicated.loc[1, "timestamp_utc"] = duplicated.loc[0, "timestamp_utc"]
        with self.assertRaisesRegex(ValueError, "unique"):
            build_causal_history_features(duplicated)


if __name__ == "__main__":
    unittest.main()
