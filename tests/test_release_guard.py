import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gridtoev.inference import PredictionService
from gridtoev.release_guard import (
    DEFAULT_REPORT_PATH,
    ReleaseAuthorizationError,
    build_release_report,
    verify_model_authorization,
)


class ReleaseGuardTests(unittest.TestCase):
    def test_v2_report_keeps_verified_v1_rollback(self) -> None:
        report = build_release_report()
        self.assertEqual(report, json.loads(DEFAULT_REPORT_PATH.read_text(encoding="utf-8")))
        self.assertEqual(report["contract_id"], "dispatch-down-benchmark-v2-purged")
        self.assertTrue(report["rollback"]["verified"])
        self.assertFalse(report["decision"]["candidate_approved"])
        self.assertIsNone(report["candidate"]["artifact_path"])

    def test_default_bundle_authorized_and_alternate_rejected(self) -> None:
        baseline = ROOT / "models" / "gridtoev_model_bundle.joblib"
        verify_model_authorization(baseline, DEFAULT_REPORT_PATH)
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.joblib"
            candidate.write_bytes(b"unapproved")
            with self.assertRaises(ReleaseAuthorizationError):
                verify_model_authorization(candidate, DEFAULT_REPORT_PATH)

    def test_tampered_baseline_fails_before_deserialization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fake = Path(directory) / "model.joblib"
            fake.write_bytes(b"tampered")
            service = PredictionService(fake, None, release_report_path=DEFAULT_REPORT_PATH)
            with self.assertRaises(ReleaseAuthorizationError):
                service.load()

    def test_loaded_default_model_exposes_release_status(self) -> None:
        service = PredictionService(
            ROOT / "models" / "gridtoev_model_bundle.joblib",
            None,
            release_report_path=DEFAULT_REPORT_PATH,
        )
        service.load()
        self.assertFalse(service.model_info()["release_status"]["candidate_approved"])


if __name__ == "__main__":
    unittest.main()
