import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gridtoev.release_preflight import (
    ReleaseAuthorizationError,
    build_release_report,
    verify_model_authorization,
)
from gridtoev.inference import PredictionService


class ReleasePreflightTests(unittest.TestCase):
    def test_current_candidate_is_rejected_and_rollback_verified(self) -> None:
        report = build_release_report()
        saved = json.loads(
            (ROOT / "benchmarks" / "release_preflight" / "release_report.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(report, saved)
        self.assertEqual(report["active_model_version"], "1.1.0")
        self.assertTrue(report["rollback"]["verified"])
        self.assertFalse(report["decision"]["candidate_approved"])
        self.assertFalse(report["guardrails"]["rolling_mae_improvement"]["passed"])
        self.assertFalse(report["guardrails"]["interval_coverage"]["passed"])
        self.assertEqual(report["candidate"]["artifact_path"], None)
        self.assertEqual(len(report["metadata"]["feature_columns"]), 119)

    def test_notebook_smoke_evidence_matches_frozen_dataset(self) -> None:
        smoke = json.loads(
            (ROOT / "benchmarks" / "release_preflight" / "notebook_smoke_report.json").read_text(
                encoding="utf-8"
            )
        )
        report = build_release_report()
        self.assertEqual(len(smoke["notebooks_executed"]), 2)
        self.assertTrue(smoke["rebuilt_dataset_matches_frozen"])
        self.assertEqual(smoke["prediction_horizons_minutes"], [30, 60])
        self.assertEqual(
            smoke["rebuilt_dataset_sha256"], report["metadata"]["dataset_sha256"]
        )

    def test_release_gate_exits_nonzero_when_candidate_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "release.json"
            process = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_release_preflight.py"),
                    "--require-candidate",
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(process.returncode, 1, process.stderr)
            self.assertTrue(output.is_file())

    def test_alternate_bundle_requires_approved_matching_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "candidate.joblib"
            candidate.write_bytes(b"candidate fixture")
            digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
            report_path = root / "release.json"
            report_path.write_text(
                json.dumps(
                    {
                        "decision": {"candidate_approved": False},
                        "candidate": {"artifact_sha256": digest},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ReleaseAuthorizationError):
                verify_model_authorization(candidate, report_path)
            report_path.write_text(
                json.dumps(
                    {
                        "decision": {"candidate_approved": True},
                        "candidate": {"artifact_sha256": "0" * 64},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ReleaseAuthorizationError, "checksum"):
                verify_model_authorization(candidate, report_path)

    def test_frozen_rollback_still_loads_without_candidate_manifest(self) -> None:
        verify_model_authorization(ROOT / "models" / "gridtoev_model_bundle.joblib", None)

    def test_default_model_reports_fallback_and_alternate_bundle_is_blocked(self) -> None:
        report_path = ROOT / "benchmarks" / "release_preflight" / "release_report.json"
        service = PredictionService(
            ROOT / "models" / "gridtoev_model_bundle.joblib",
            None,
            release_report_path=report_path,
        )
        service.load()
        self.assertFalse(service.model_info()["release_status"]["candidate_approved"])
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.joblib"
            candidate.write_bytes(b"unapproved")
            blocked = PredictionService(candidate, None, release_report_path=report_path)
            with self.assertRaises(ReleaseAuthorizationError):
                blocked.load()


if __name__ == "__main__":
    unittest.main()
