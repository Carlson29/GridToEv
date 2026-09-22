import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gridtoev.inference import PredictionService


class RealArtifactTests(unittest.TestCase):
    def test_committed_bundle_predicts_both_horizons(self) -> None:
        service = PredictionService(
            ROOT / "models" / "gridtoev_model_bundle.joblib",
            ROOT / "data" / "processed" / "gridtoev_model_ready.csv",
        )
        service.load()
        predictions = service.predict_latest(100.0)
        self.assertEqual({prediction["forecast_horizon_minutes"] for prediction in predictions}, {30, 60})
        for prediction in predictions:
            self.assertGreaterEqual(prediction["dispatch_down_probability"], 0.0)
            self.assertLessEqual(prediction["dispatch_down_probability"], 1.0)
            self.assertGreaterEqual(prediction["predicted_dispatch_down_mwh"], 0.0)


if __name__ == "__main__":
    unittest.main()
