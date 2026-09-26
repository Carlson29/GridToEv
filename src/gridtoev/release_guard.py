"""Fail-closed artifact authorization for the shareable prediction service.

The v2 benchmark is an evaluation contract, not approval of a new model.  Until
a candidate passes a separate release gate, only the pinned v1.1.0 bytes serve.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .constants import PROJECT_ROOT


DEFAULT_REPORT_PATH = PROJECT_ROOT / "benchmarks" / "release_preflight" / "release_report.v2.json"
CONTRACT_PATH = PROJECT_ROOT / "config" / "benchmark_contract.v2.json"


class ReleaseAuthorizationError(ValueError):
    """The requested model bytes are not authorized for prediction."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_contract_sha256(columns: list[str]) -> str:
    encoded = json.dumps(columns, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_release_report(root: Path | str = PROJECT_ROOT) -> dict[str, Any]:
    """Record the active rollback and the absence of an approved candidate."""
    root = Path(root).resolve()
    contract = json.loads((root / "config" / "benchmark_contract.v2.json").read_text(encoding="utf-8"))
    benchmark = json.loads((root / "benchmarks" / "v2-purged" / "benchmark_report.json").read_text(encoding="utf-8"))
    metadata = json.loads((root / "models" / "model_metadata.json").read_text(encoding="utf-8"))
    frozen = contract["frozen_baseline"]
    checks = {}
    for name in ("dataset", "model_artifact", "metrics", "metadata"):
        path = root / frozen[f"{name}_path"]
        expected = frozen[f"{name}_sha256"]
        observed = sha256_file(path)
        checks[name] = {"path": frozen[f"{name}_path"], "sha256": observed, "passed": observed == expected}
    verified = all(item["passed"] for item in checks.values())
    if benchmark["contract"]["contract_id"] != contract["contract_id"]:
        raise ReleaseAuthorizationError("Purged benchmark and release contract disagree")
    if not benchmark["frozen_baseline_verification"]["passed"]:
        raise ReleaseAuthorizationError("Purged benchmark rollback verification failed")
    return {
        "schema_version": 2,
        "contract_id": contract["contract_id"],
        "active_model_version": metadata["model_version"],
        "decision": {
            "candidate_approved": False,
            "reason": "No new candidate passed the v2 purged rolling, final, interval and live-feature gates; keep v1.1.0.",
        },
        "candidate": {"artifact_path": None, "artifact_sha256": None},
        "rollback": {"verified": verified, "checks": checks},
        "evaluation": {
            "rolling_mae_mwh": benchmark["rolling_origin"]["aggregate"]["overall"]["mae"],
            "final_test_mae_mwh": benchmark["final_test"]["dispatch_breakdown"]["overall"]["mae"],
            "release_thresholds": contract["release_thresholds"],
        },
        "feature_contract": {
            "count": len(metadata["feature_columns"]),
            "sha256": feature_contract_sha256(metadata["feature_columns"]),
        },
    }


def verify_model_authorization(model_path: Path | str, report_path: Path | str) -> None:
    """Check bytes before joblib deserialization; an alternate path needs approval."""
    model_path = Path(model_path).resolve()
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    contract_path = Path(os.getenv("GRIDTOEV_CONTRACT_PATH", str(CONTRACT_PATH))).resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    project_root = contract_path.parent.parent
    if report.get("contract_id") != contract["contract_id"]:
        raise ReleaseAuthorizationError("Release report does not match the v2 contract")
    if not report.get("rollback", {}).get("verified", False):
        raise ReleaseAuthorizationError("Release report rollback verification failed")
    expected_baseline = (project_root / contract["frozen_baseline"]["model_artifact_path"]).resolve()
    if model_path == expected_baseline:
        expected_hash = contract["frozen_baseline"]["model_artifact_sha256"]
    else:
        candidate = report.get("candidate", {})
        if not report.get("decision", {}).get("candidate_approved"):
            raise ReleaseAuthorizationError("Alternate model has not passed release gates")
        approved_path = candidate.get("artifact_path")
        if not approved_path or model_path != (project_root / approved_path).resolve():
            raise ReleaseAuthorizationError("Alternate model path is not approved")
        expected_hash = candidate.get("artifact_sha256")
    if not expected_hash or not model_path.exists() or sha256_file(model_path) != expected_hash:
        raise ReleaseAuthorizationError("Model artifact checksum mismatch")
