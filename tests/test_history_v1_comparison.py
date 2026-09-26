import unittest
import json
from pathlib import Path

import pandas as pd

from gridtoev.history_v1_comparison import align_history_to_v1, is_serving_eligible


class HistoryV1ComparisonTests(unittest.TestCase):
    def test_committed_same_row_report_is_gated_and_reproducible(self) -> None:
        root = Path(__file__).resolve().parents[1]
        report = json.loads(
            (root / "benchmarks/optional_v2/multi_year_same_rows.json").read_text(encoding="utf-8")
        )
        contract = json.loads(
            (root / "config/benchmark_contract.v2.json").read_text(encoding="utf-8")
        )
        self.assertAlmostEqual(
            report["development"]["v1_mae_mwh"],
            contract["frozen_baseline"]["rolling_mae_mwh"],
        )
        self.assertGreater(
            report["development"]["candidate_mae_mwh"],
            report["development"]["v1_mae_mwh"],
        )
        self.assertFalse(report["sealed_final_scored"])
        self.assertFalse(report["serving_eligible"])
        self.assertFalse(report["artifact_created"])

    def setUp(self) -> None:
        self.v1 = pd.DataFrame(
            {
                "issue_timestamp_utc": ["2026-01-02T00:00:00Z"],
                "target_timestamp_utc": ["2026-01-02T00:30:00Z"],
                "forecast_horizon_minutes": [30],
                "dispatch_down_mwh": [12.0],
            }
        )
        self.history = self.v1.copy()
        self.history["feature_timestamp_utc"] = "2026-01-01T00:00:00Z"
        self.history["latest_eligible_observation_utc"] = "2026-01-01T00:00:00Z"

    def test_alignment_requires_same_rows_labels_and_cutoff(self) -> None:
        result = align_history_to_v1(self.v1, self.history, delay_hours=24)
        self.assertEqual(result["dispatch_down_mwh"].tolist(), [12.0])
        self.assertEqual(len(result), 1)

        late = self.history.copy()
        late["feature_timestamp_utc"] = "2026-01-01T00:30:00Z"
        with self.assertRaisesRegex(ValueError, "cutoff"):
            align_history_to_v1(self.v1, late, delay_hours=24)

        changed = self.history.copy()
        changed["dispatch_down_mwh"] = 99.0
        with self.assertRaisesRegex(ValueError, "label"):
            align_history_to_v1(self.v1, changed, delay_hours=24)

    def test_unverified_publication_blocks_serving_even_if_mae_is_lower(self) -> None:
        self.assertFalse(is_serving_eligible(9.0, 10.0, publication_verified=False, final_scored=True))
        self.assertFalse(is_serving_eligible(9.0, 10.0, publication_verified=True, final_scored=False))
        self.assertFalse(is_serving_eligible(10.0, 10.0, publication_verified=True, final_scored=True))
        self.assertTrue(is_serving_eligible(9.0, 10.0, publication_verified=True, final_scored=True))


if __name__ == "__main__":
    unittest.main()
