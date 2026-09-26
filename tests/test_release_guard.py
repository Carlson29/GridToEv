import json
import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


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

    def test_unverified_rollback_report_rejects_even_pinned_model(self) -> None:
        baseline = ROOT / "models" / "gridtoev_model_bundle.joblib"
        report = json.loads(DEFAULT_REPORT_PATH.read_text(encoding="utf-8"))
        report["rollback"]["verified"] = False
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "release.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ReleaseAuthorizationError, "rollback"):
                verify_model_authorization(baseline, report_path)

    def test_report_feature_hash_must_match_loaded_bundle(self) -> None:
        baseline = ROOT / "models" / "gridtoev_model_bundle.joblib"
        report = json.loads(DEFAULT_REPORT_PATH.read_text(encoding="utf-8"))
        report["feature_contract"]["sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "release.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            service = PredictionService(baseline, None, release_report_path=report_path)
            with self.assertRaisesRegex(ReleaseAuthorizationError, "feature contract"):
                service.load()

    def test_container_contract_override_authorizes_baseline_under_app_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config").mkdir()
            (root / "models").mkdir()
            model = root / "models" / "gridtoev_model_bundle.joblib"
            model.write_bytes(b"test pinned bytes")
            contract = json.loads((ROOT / "config" / "benchmark_contract.v2.json").read_text(encoding="utf-8"))
            contract["frozen_baseline"]["model_artifact_sha256"] = hashlib.sha256(model.read_bytes()).hexdigest()
            contract_path = root / "config" / "benchmark_contract.v2.json"
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            report = json.loads(DEFAULT_REPORT_PATH.read_text(encoding="utf-8"))
            report_path = root / "release.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with patch.dict(os.environ, {"GRIDTOEV_CONTRACT_PATH": str(contract_path)}):
                verify_model_authorization(model, report_path)

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
