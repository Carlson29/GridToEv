import inspect
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gridtoev.calibration import (
    apply_interval_policy,
    assess_component_refit,
    fit_interval_policy,
    reconcile_components,
    select_interval_policy,
)


def oof_fixture() -> pd.DataFrame:
    times = pd.date_range("2026-01-01", periods=24, freq="30min", tz="UTC")
    rows = []
    for index, timestamp in enumerate(times):
        for horizon in (30, 60):
            center = 20.0 + index / 10
            residual = (-4.0 if index % 2 else 6.0) * (horizon / 30)
            rows.append(
                {
                    "issue_timestamp_utc": timestamp,
                    "forecast_horizon_minutes": horizon,
                    "fold": f"fold_{index // 6}",
                    "dispatch_down_mwh": center + residual,
                    "v1.1.0": center,
                }
            )
    return pd.DataFrame(rows)


class CalibrationTests(unittest.TestCase):
    def test_horizon_policy_uses_only_supplied_oof_rows(self) -> None:
        oof = oof_fixture()
        policy = fit_interval_policy(oof, "horizon_symmetric")
        self.assertEqual(set(policy.offsets_by_horizon), {"30", "60"})
        self.assertGreater(policy.offsets_by_horizon["60"]["upper"],
                           policy.offsets_by_horizon["30"]["upper"])
        self.assertEqual(policy.calibration_rows, len(oof))
        self.assertNotIn("final_test", inspect.signature(select_interval_policy).parameters)

    def test_final_test_fold_is_rejected_during_selection(self) -> None:
        oof = oof_fixture()
        oof.loc[0, "fold"] = "final_test"
        with self.assertRaisesRegex(ValueError, "final.test"):
            select_interval_policy(oof)

    def test_intervals_are_ordered_nonnegative_and_contain_p50(self) -> None:
        policy = fit_interval_policy(oof_fixture(), "horizon_asymmetric")
        horizons = np.array([30, 60, 30])
        p10, p50, p90 = apply_interval_policy(
            horizons, np.array([0.0, 2.0, 50.0]), policy
        )
        self.assertTrue(np.all(p10 >= 0))
        self.assertTrue(np.all(p10 <= p50))
        self.assertTrue(np.all(p50 <= p90))

    def test_component_reconciliation_handles_zero_and_negative_raw_values(self) -> None:
        total = np.array([0.0, 10.0, 7.0, 8.0])
        raw_curtailment = np.array([10.0, 3.0, -4.0, 0.0])
        raw_constraint = np.array([4.0, 7.0, -3.0, 0.0])
        curtailment, constraint = reconcile_components(
            total, raw_curtailment, raw_constraint, default_curtailment_share=0.3
        )
        np.testing.assert_allclose(curtailment + constraint, total)
        self.assertTrue(np.all(curtailment >= 0))
        self.assertTrue(np.all(constraint >= 0))
        self.assertAlmostEqual(curtailment[-1], 2.4)

    def test_component_refit_requires_longer_history(self) -> None:
        frame = oof_fixture().rename(columns={"dispatch_down_mwh": "curtailment_mwh"})
        frame["constraint_mwh"] = frame["curtailment_mwh"]
        result = assess_component_refit(frame)
        self.assertFalse(result["eligible"])
        self.assertIn("history_days", result["failed_checks"])


if __name__ == "__main__":
    unittest.main()
