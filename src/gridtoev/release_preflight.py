"""Evidence-based release gate for the frozen GridToEV model contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .constants import DEFAULT_MODEL_PATH, PROJECT_ROOT


DEFAULT_REPORT_PATH = PROJECT_ROOT / "benchmarks" / "release_preflight" / "release_report.json"


class ReleaseAuthorizationError(ValueError):
    """A model artifact is not approved by the release report."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_contract_sha256(columns: list[str]) -> str:
    encoded = json.dumps(columns, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _gate(observed: Any, target: Any, passed: bool) -> dict[str, Any]:
    return {"observed": observed, "target": target, "passed": bool(passed)}


def build_release_report(root: Path | str = PROJECT_ROOT) -> dict[str, Any]:
    """Evaluate every release threshold without creating a candidate model."""
    root = Path(root).resolve()
    contract = _json(root / "config" / "benchmark_contract.v1.json")
    baseline = _json(root / "benchmarks" / "v1.1.0" / "benchmark_report.json")
    horizon = _json(root / "benchmarks" / "horizon_models" / "benchmark_report.json")
    calibration = _json(root / "benchmarks" / "calibrated_intervals" / "calibration_report.json")
    metadata = _json(root / "models" / "model_metadata.json")
    quality = _json(root / "data" / "processed" / "gridtoev_quality_report.json")

    frozen = contract["frozen_baseline"]
    rollback_checks = {
        "dataset_sha256": (root / frozen["dataset_path"], frozen["dataset_sha256"]),
        "model_artifact_sha256": (
            root / frozen["model_artifact_path"], frozen["model_artifact_sha256"]
        ),
        "metrics_sha256": (root / frozen["metrics_path"], frozen["metrics_sha256"]),
        "metadata_sha256": (root / frozen["metadata_path"], frozen["metadata_sha256"]),
    }
    verified_hashes: dict[str, dict[str, Any]] = {}
    for name, (path, expected) in rollback_checks.items():
        actual = sha256_file(path)
        verified_hashes[name] = {
            "path": path.relative_to(root).as_posix(),
            "expected": expected,
            "actual": actual,
            "passed": actual == expected,
        }
    rollback_verified = all(item["passed"] for item in verified_hashes.values())

    if horizon["contract"]["contract_id"] != contract["contract_id"]:
        raise ValueError("Issue #8 uses a different benchmark contract")
    if calibration["contract_id"] != contract["contract_id"]:
        raise ValueError("Issue #9 uses a different benchmark contract")
    if horizon["dataset"]["sha256"] != frozen["dataset_sha256"]:
        raise ValueError("Issue #8 model comparison used a different dataset")
    if calibration["dataset_sha256"] != frozen["dataset_sha256"]:
        raise ValueError("Issue #9 calibration used a different dataset")

    threshold = contract["release_thresholds"]
    development = horizon["selection"]["development_gate"]
    final = horizon["final_test"]
    interval = calibration["final_test"]["candidate"]
    has_candidate_final = bool(final["accessed"] and final["metrics"])
    final_metrics = final["metrics"]["overall"] if has_candidate_final else None
    candidate_artifact_path: str | None = None
    candidate_artifact_sha256: str | None = None

    guardrails = {
        "rolling_mae_improvement": _gate(
            development["aggregate_mae_improvement_fraction"],
            threshold["rolling_mae_improvement_min"],
            development["checks"]["aggregate_improvement"],
        ),
        "both_horizons": _gate(
            development["horizon_mae_improvement_fraction"],
            "neither horizon regresses",
            development["checks"]["horizon_non_regression"],
        ),
        "fold_stability": _gate(
            development["fold_mae_improvement_fraction"],
            "at least 3 of 4 folds improve; worst fold no more than 2% worse",
            development["checks"]["fold_stability"],
        ),
        "final_test_mae": _gate(
            final_metrics["mae"] if final_metrics else None,
            threshold["final_test_mae_mwh_max"],
            bool(final_metrics and final_metrics["mae"] <= threshold["final_test_mae_mwh_max"]),
        ),
        "final_test_rmse": _gate(
            final_metrics["rmse"] if final_metrics else None,
            threshold["final_test_rmse_mwh_max"],
            bool(final_metrics and final_metrics["rmse"] <= threshold["final_test_rmse_mwh_max"]),
        ),
        "final_test_wape": _gate(
            final_metrics["wape"] if final_metrics else None,
            threshold["final_test_wape_max"],
            bool(final_metrics and final_metrics["wape"] <= threshold["final_test_wape_max"]),
        ),
        "p50_pinball": _gate(
            interval["p50_pinball_loss"],
            threshold["final_test_p50_pinball_loss_max"],
            calibration["release_checks"]["p50_pinball"],
        ),
        "interval_coverage": _gate(
            interval["coverage"],
            [
                threshold["final_test_p10_p90_coverage_min"],
                threshold["final_test_p10_p90_coverage_max"],
            ],
            calibration["release_checks"]["coverage_min"]
            and calibration["release_checks"]["coverage_max"],
        ),
        "interval_width": _gate(
            calibration["final_test"]["average_width_ratio_to_deployed_v1.1.0"],
            "no more than 1.25 times deployed v1.1.0 interval width",
            calibration["release_checks"]["width_not_excessive"],
        ),
        "quantile_order": _gate(
            calibration["release_checks"]["quantiles_ordered_nonnegative"],
            True,
            calibration["release_checks"]["quantiles_ordered_nonnegative"],
        ),
        "component_reconciliation": _gate(
            calibration["components"]["reconciliation_max_absolute_error_mwh"],
            "<= 1e-9 MWh",
            calibration["release_checks"]["components_reconciled"],
        ),
        "candidate_artifact": _gate(None, "versioned candidate bundle available", False),
    }
    candidate_approved = rollback_verified and all(
        item["passed"] for item in guardrails.values()
    )
    source_manifest = root / "data" / "processed" / "source_manifest.csv"
    return {
        "schema_version": 1,
        "issue": 10,
        "contract_id": contract["contract_id"],
        "active_model_version": metadata["model_version"],
        "decision": {
            "candidate_approved": candidate_approved,
            "active_artifact": frozen["model_artifact_path"],
            "reason": (
                "No candidate passed the rolling model gate and calibrated interval coverage. "
                "Keep the frozen v1.1.0 bundle active."
            ),
        },
        "candidate": {
            "name": horizon["selection"]["development_winner"],
            "artifact_path": candidate_artifact_path,
            "artifact_sha256": candidate_artifact_sha256,
            "development_gate_passed": development["passed"],
            "interval_gate_passed": calibration["release_candidate_eligible"],
            "final_test_scored": has_candidate_final,
        },
        "rollback": {
            "model_version": contract["baseline_model_version"],
            "verified": rollback_verified,
            "hashes": verified_hashes,
        },
        "guardrails": guardrails,
        "metadata": {
            "dataset_sha256": frozen["dataset_sha256"],
            "source_manifest_sha256": sha256_file(source_manifest),
            "source_coverage": quality,
            "feature_columns": metadata["feature_columns"],
            "feature_contract_sha256": feature_contract_sha256(metadata["feature_columns"]),
            "forecast_horizons_minutes": metadata["forecast_horizons_minutes"],
            "partitions": metadata["partitions"],
            "rolling_origin_windows": contract["rolling_origin"]["train_score_windows"],
            "operating_policy": baseline["selection"]["operating_policy"],
            "training_config": metadata["training_config"],
            "library_versions": metadata["library_versions"],
            "upstream_feature_decisions": horizon["upstream_feature_decisions"],
        },
    }


def verify_model_authorization(
    model_path: Path | str,
    release_report_path: Path | str | None,
    *,
    baseline_path: Path | str = DEFAULT_MODEL_PATH,
) -> None:
    """Allow the known rollback or an approved candidate with matching bytes."""
    model_path = Path(model_path).resolve()
    baseline_path = Path(baseline_path).resolve()
    if model_path == baseline_path:
        contract = _json(PROJECT_ROOT / "config" / "benchmark_contract.v1.json")
        expected = contract["frozen_baseline"]["model_artifact_sha256"]
        if not model_path.exists() or sha256_file(model_path) != expected:
            raise ReleaseAuthorizationError("Frozen rollback model checksum mismatch")
        return
    if release_report_path is None or not Path(release_report_path).exists():
        raise ReleaseAuthorizationError("Alternate model requires an approved release report")
    report = _json(Path(release_report_path))
    if not report.get("decision", {}).get("candidate_approved", False):
        raise ReleaseAuthorizationError("Candidate model has not passed the release guardrails")
    expected = report.get("candidate", {}).get("artifact_sha256")
    if not expected or not model_path.exists() or sha256_file(model_path) != expected:
        raise ReleaseAuthorizationError("Candidate model checksum does not match release report")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GridToEV model release guardrails")
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--require-candidate",
        action="store_true",
        help="Exit nonzero unless a replacement candidate passes every release gate",
    )
    args = parser.parse_args()
    report = build_release_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Rollback verified: {report['rollback']['verified']}")
    print(f"Candidate approved: {report['decision']['candidate_approved']}")
    print(f"Active model: {report['active_model_version']}")
    print(f"Report: {args.output}")
    if args.require_candidate and not report["decision"]["candidate_approved"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
