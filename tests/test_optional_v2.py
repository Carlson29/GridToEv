import unittest

import numpy as np
import pandas as pd

from gridtoev.optional_v2_experiment import FamilyScore, _eligible, is_final_artifact_eligible
from gridtoev.optional_v2 import (
    HorizonResidualModel,
    blend_predictions,
    candidate_features,
    select_blend_weights,
)


class OptionalV2Tests(unittest.TestCase):
    def test_physical_features_use_only_supplied_issue_row(self) -> None:
        frame = pd.DataFrame(
            {
                "eirgrid_ie_wind_availability_mw": [120.0],
                "eirgrid_ie_wind_generation_mw": [100.0],
                "wind_lag_1": [90.0],
                "wind_lag_2": [85.0],
                "eirgrid_ie_demand_mw": [500.0],
                "load_lag_1": [490.0],
                "load_lag_2": [485.0],
                "eirgrid_snsp_ratio": [0.72],
            }
        )
        features = candidate_features(frame, physical=True)
        self.assertEqual(features.loc[0, "v2_wind_unrealised_mw"], 20.0)
        self.assertEqual(features.loc[0, "v2_wind_acceleration_mw"], 5.0)
        self.assertEqual(features.loc[0, "v2_load_acceleration_mw"], 5.0)
        self.assertAlmostEqual(features.loc[0, "v2_snsp_headroom_ratio"], 0.03)
        self.assertFalse(any("target" in name for name in features))

    def test_horizon_model_and_blend_keep_horizons_separate(self) -> None:
        frame = pd.DataFrame(
            {
                "forecast_horizon_minutes": [30, 30, 30, 30, 60, 60, 60, 60],
                "dispatch_down_mwh_latest_observed": [2, 3, 4, 5, 10, 11, 12, 13],
                "feature": [1, 2, 3, 4, 1, 2, 3, 4],
                "dispatch_down_mwh": [2, 3, 4, 5, 10, 11, 12, 13],
            }
        )
        model = HorizonResidualModel(["feature"], n_estimators=8, min_samples_leaf=1)
        model.fit(frame)
        predicted = model.predict(frame)
        self.assertEqual(len(predicted), len(frame))
        self.assertTrue(np.isfinite(predicted).all())
        blended = blend_predictions(frame, np.ones(8), np.full(8, 3.0), {"30": 0.0, "60": 1.0})
        self.assertEqual(blended.tolist(), [1.0] * 4 + [3.0] * 4)

    def test_weight_selection_uses_only_development_labels(self) -> None:
        frame = pd.DataFrame(
            {
                "forecast_horizon_minutes": [30, 30, 60, 60],
                "dispatch_down_mwh": [5.0, 5.0, 8.0, 8.0],
            }
        )
        weights = select_blend_weights(
            frame,
            baseline=np.array([1.0, 2.0, 2.0, 3.0]),
            candidate=np.array([5.0, 5.0, 8.0, 8.0]),
        )
        self.assertEqual(weights, {"30": 0.5, "60": 0.5})

        restricted = select_blend_weights(
            frame,
            baseline=np.array([1.0, 2.0, 2.0, 3.0]),
            candidate=np.array([5.0, 5.0, 8.0, 8.0]),
            weight_grid=[0.0, 0.2],
        )
        self.assertEqual(restricted, {"30": 0.2, "60": 0.2})

    def test_development_gate_rejects_insufficient_gain(self) -> None:
        score = FamilyScore(
            name="test", physical=False, weights={"30": 0.2, "60": 0.2},
            baseline_mae=10.0, candidate_mae=9.9,
            by_horizon={"30": {"v1_mae_mwh": 10.0, "v2_mae_mwh": 9.9},
                        "60": {"v1_mae_mwh": 10.0, "v2_mae_mwh": 9.9}},
            by_fold={},
        )
        self.assertTrue(_eligible(score, 0.03, 0.0))
        self.assertFalse(_eligible(score, 0.03, 0.02))

    def test_final_artifact_requires_verified_asof_features(self) -> None:
        self.assertFalse(is_final_artifact_eligible(9.0, 10.0, 0.0, publication_verified=False))
        self.assertFalse(is_final_artifact_eligible(10.0, 10.0, 0.0, publication_verified=True))
        self.assertTrue(is_final_artifact_eligible(9.0, 10.0, 0.0, publication_verified=True))


if __name__ == "__main__":
    unittest.main()
