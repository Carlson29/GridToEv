from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from .constants import (
    DEFAULT_DATASET_PATH,
    MODEL_VERSION,
    PROJECT_ROOT,
    ROLLING_ORIGIN_WINDOWS,
)
from .training import (
    OperatingPolicy,
    TrainingConfig,
    _blend_dispatch_predictions,
    _build_models,
    _dispatch_ml_prediction,
    _fit_models,
    _partition_metadata,
    _regression_metrics,
    _regressor,
    _score_models,
    _select_operating_policy,
    chronological_split,
    feature_columns,
    load_dataset,
)


DEFAULT_CONTRACT_PATH = PROJECT_ROOT / "config" / "benchmark_contract.v1.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "benchmarks" / "v1.1.0"


class BenchmarkDataError(ValueError):
    """Raised when a dataset violates the benchmark's chronological contract."""


class BenchmarkVerificationError(RuntimeError):
    """Raised when the committed baseline cannot be reproduced exactly enough."""


@dataclass(frozen=True)
class BenchmarkContract:
    schema_version: int
    contract_id: str
    baseline_model_version: str
    train_fraction: float
    validation_fraction: float
    final_test_fraction: float
    final_test_policy: str
    rolling_origin_windows: tuple[tuple[float, float], ...]
    include_validation_fold: bool
    release_thresholds: dict[str, float]
    frozen_baseline: dict[str, Any]


@dataclass(frozen=True)
class RollingOriginFold:
    name: str
    fit: pd.DataFrame
    score: pd.DataFrame


@dataclass(frozen=True)
class BenchmarkResult:
    report: dict[str, Any]
    report_path: Path
    registry_path: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_benchmark_contract(path: Path | str = DEFAULT_CONTRACT_PATH) -> BenchmarkContract:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    split = payload["split"]
    rolling = payload["rolling_origin"]
    windows = tuple(
        (float(train_end), float(score_end))
        for train_end, score_end in rolling["train_score_windows"]
    )
    if any(not 0 < train_end < score_end <= 1 for train_end, score_end in windows):
        raise ValueError("Rolling-origin windows must satisfy 0 < train_end < score_end <= 1")
    if windows != ROLLING_ORIGIN_WINDOWS:
        raise ValueError(
            "Benchmark contract rolling-origin windows do not match the training implementation"
        )
    if abs(
        float(split["train_fraction"])
        + float(split["validation_fraction"])
        + float(split["final_test_fraction"])
        - 1.0
    ) > 1e-12:
        raise ValueError("Benchmark split fractions must sum to one")
    return BenchmarkContract(
        schema_version=int(payload["schema_version"]),
        contract_id=str(payload["contract_id"]),
        baseline_model_version=str(payload["baseline_model_version"]),
        train_fraction=float(split["train_fraction"]),
        validation_fraction=float(split["validation_fraction"]),
        final_test_fraction=float(split["final_test_fraction"]),
        final_test_policy=str(split["final_test_policy"]),
        rolling_origin_windows=windows,
        include_validation_fold=bool(rolling["include_validation_fold"]),
        release_thresholds={
            key: float(value) for key, value in payload["release_thresholds"].items()
        },
        frozen_baseline=dict(payload["frozen_baseline"]),
    )


def validate_benchmark_dataset(path: Path | str) -> None:
    """Fail closed on ordering, horizon alignment, duplicates, or future publications."""
    path = Path(path)
    frame = pd.read_csv(path)
    required = {
        "issue_timestamp_utc",
        "target_timestamp_utc",
        "forecast_horizon_minutes",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise BenchmarkDataError(f"Benchmark dataset is missing required columns: {missing}")

    issue_time = pd.to_datetime(frame["issue_timestamp_utc"], errors="raise", utc=True)
    target_time = pd.to_datetime(frame["target_timestamp_utc"], errors="raise", utc=True)
    horizon = pd.to_numeric(frame["forecast_horizon_minutes"], errors="raise")
    keys = pd.DataFrame(
        {
            "issue_timestamp_utc": issue_time,
            "forecast_horizon_minutes": horizon,
        }
    )
    sorted_keys = keys.sort_values(
        ["issue_timestamp_utc", "forecast_horizon_minutes"], kind="stable"
    ).reset_index(drop=True)
    if not keys.reset_index(drop=True).equals(sorted_keys):
        raise BenchmarkDataError(
            "Benchmark dataset must be chronologically sorted by issue time and horizon"
        )
    if keys.duplicated().any():
        raise BenchmarkDataError("Benchmark dataset contains duplicate issue-time/horizon keys")

    expected_target = issue_time + pd.to_timedelta(horizon, unit="m")
    if (target_time <= issue_time).any() or not target_time.equals(expected_target):
        raise BenchmarkDataError(
            "Target timestamps must be strictly future and exactly match the forecast horizon"
        )

    publication_columns = [
        column
        for column in frame.columns
        if column in {"published_at_utc", "source_vintage_utc"}
        or column.endswith("_published_at_utc")
        or column.endswith("_source_vintage_utc")
    ]
    for column in publication_columns:
        raw = frame[column]
        present = raw.notna() & raw.astype(str).str.strip().ne("")
        published = pd.to_datetime(raw, errors="coerce", utc=True)
        if published.loc[present].isna().any():
            raise BenchmarkDataError(f"Publication column {column} contains invalid timestamps")
        if (published.loc[present] > issue_time.loc[present]).any():
            raise BenchmarkDataError(
                f"Feature in {column} was published after issue_timestamp_utc"
            )


def build_rolling_origin_folds(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    contract: BenchmarkContract,
) -> list[RollingOriginFold]:
    """Build the versioned expanding-window folds without including final-test rows."""
    unique_times = np.array(sorted(train["issue_timestamp_utc"].unique()))
    folds: list[RollingOriginFold] = []
    for index, (train_end_fraction, score_end_fraction) in enumerate(
        contract.rolling_origin_windows,
        start=1,
    ):
        train_end = max(1, int(len(unique_times) * train_end_fraction))
        score_end = max(train_end + 1, int(len(unique_times) * score_end_fraction))
        score_end = min(score_end, len(unique_times))
        fit_times = set(unique_times[:train_end])
        score_times = set(unique_times[train_end:score_end])
        fit = train.loc[train["issue_timestamp_utc"].isin(fit_times)].copy()
        score = train.loc[train["issue_timestamp_utc"].isin(score_times)].copy()
        if fit.empty or score.empty:
            raise ValueError(f"Rolling-origin fold {index} produced an empty partition")
        folds.append(RollingOriginFold(f"train_fold_{index}", fit, score))

    if contract.include_validation_fold:
        folds.append(RollingOriginFold("validation_fold", train.copy(), validation.copy()))

    for fold in folds:
        if fold.fit["issue_timestamp_utc"].max() >= fold.score["issue_timestamp_utc"].min():
            raise ValueError(f"Rolling-origin fold {fold.name} crosses its score boundary")
    return folds


def _dispatch_breakdown(
    frame: pd.DataFrame,
    predictions: np.ndarray,
) -> dict[str, Any]:
    truth = frame["dispatch_down_mwh"]
    result: dict[str, Any] = {
        "overall": _regression_metrics(truth, predictions),
        "by_horizon": {},
        "by_event_regime": {},
    }
    for horizon in sorted(frame["forecast_horizon_minutes"].astype(int).unique()):
        mask = frame["forecast_horizon_minutes"].eq(horizon).to_numpy()
        result["by_horizon"][str(horizon)] = _regression_metrics(
            truth.loc[mask], predictions[mask]
        )
    regimes = {
        "no_dispatch_down": frame["dispatch_down_event"].eq(0).to_numpy(),
        "positive_dispatch_down": frame["dispatch_down_event"].eq(1).to_numpy(),
    }
    for name, mask in regimes.items():
        if mask.any():
            result["by_event_regime"][name] = _regression_metrics(
                truth.loc[mask], predictions[mask]
            )
    return result


def _fold_predictions(
    fold: RollingOriginFold,
    columns: list[str],
    policy: OperatingPolicy,
    config: TrainingConfig,
) -> np.ndarray:
    model = _regressor(config, loss="squared_error")
    residual = (
        fold.fit["dispatch_down_mwh"]
        - fold.fit["dispatch_down_mwh_latest_observed"]
    )
    model.fit(fold.fit[columns], residual)
    latest = fold.score["dispatch_down_mwh_latest_observed"].to_numpy(dtype=float)
    ml_prediction = np.maximum(latest + model.predict(fold.score[columns]), 0.0)
    blended, _ = _blend_dispatch_predictions(
        fold.score,
        ml_prediction,
        policy.dispatch_trend_alpha_by_horizon,
        policy.dispatch_ml_weight_by_horizon,
    )
    return blended


def _fold_report(
    fold: RollingOriginFold,
    columns: list[str],
    policy: OperatingPolicy,
    config: TrainingConfig,
) -> tuple[dict[str, Any], np.ndarray]:
    predictions = _fold_predictions(fold, columns, policy, config)
    report = {
        "name": fold.name,
        "fit": _partition_metadata(fold.fit),
        "score": _partition_metadata(fold.score),
        "dispatch_breakdown": _dispatch_breakdown(fold.score, predictions),
    }
    return report, predictions


def _verify_frozen_baseline(
    contract: BenchmarkContract,
    dataset_path: Path,
    metrics: dict[str, Any],
    final_test: pd.DataFrame,
) -> None:
    frozen = contract.frozen_baseline
    checksums = {
        "dataset_sha256": _sha256(dataset_path),
        "model_artifact_sha256": _sha256(PROJECT_ROOT / frozen["model_artifact_path"]),
        "metrics_sha256": _sha256(PROJECT_ROOT / frozen["metrics_path"]),
        "metadata_sha256": _sha256(PROJECT_ROOT / frozen["metadata_path"]),
    }
    for key, actual in checksums.items():
        expected = str(frozen[key]).lower()
        if actual.lower() != expected:
            raise BenchmarkVerificationError(
                f"Frozen baseline {key} mismatch: expected {expected}, got {actual}"
            )

    expected_period = frozen["final_test_period"]
    actual_period = _partition_metadata(final_test)
    if actual_period != expected_period:
        raise BenchmarkVerificationError(
            f"Final-test period mismatch: expected {expected_period}, got {actual_period}"
        )

    actual_metrics = {
        "dispatch_down_mae_mwh": metrics["regression"]["dispatch_down_mwh"]["mae"],
        "dispatch_down_rmse_mwh": metrics["regression"]["dispatch_down_mwh"]["rmse"],
        "dispatch_down_wape": metrics["regression"]["dispatch_down_mwh"]["wape"],
        "event_average_precision": metrics["classification"]["test"]["average_precision"],
        "p50_pinball_loss": metrics["uncertainty"]["p50_pinball_loss"],
        "p10_p90_empirical_coverage": metrics["uncertainty"][
            "p10_p90_empirical_coverage"
        ],
    }
    tolerance = float(frozen["numeric_tolerance"])
    for key, expected in frozen["metrics"].items():
        actual = actual_metrics[key]
        if not np.isclose(actual, float(expected), rtol=0.0, atol=tolerance):
            raise BenchmarkVerificationError(
                f"Frozen baseline metric {key} mismatch: expected {expected}, got {actual}"
            )


def _upsert_experiment_registry(path: Path, row: dict[str, Any]) -> None:
    """Replace the same run while preserving candidate rows written by later work."""
    current = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if "run_id" in current.columns:
        current = current.loc[current["run_id"] != row["run_id"]].copy()
    updated = pd.concat([current, pd.DataFrame([row])], ignore_index=True, sort=False)
    updated.to_csv(path, index=False)


def run_baseline_benchmark(
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
    contract_path: Path | str = DEFAULT_CONTRACT_PATH,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    training_config: TrainingConfig | None = None,
    verify_frozen_baseline: bool = True,
) -> BenchmarkResult:
    """Reproduce the baseline without writing or replacing the production model artifact."""
    started = perf_counter()
    dataset_path = Path(dataset_path).resolve()
    output_dir = Path(output_dir).resolve()
    contract = load_benchmark_contract(contract_path)
    config = training_config or TrainingConfig(
        train_fraction=contract.train_fraction,
        validation_fraction=contract.validation_fraction,
    )
    if (
        config.train_fraction != contract.train_fraction
        or config.validation_fraction != contract.validation_fraction
    ):
        raise ValueError("TrainingConfig split fractions must match the benchmark contract")

    validate_benchmark_dataset(dataset_path)
    data = load_dataset(dataset_path)
    columns = feature_columns(data)
    train, validation, final_test = chronological_split(
        data,
        train_fraction=contract.train_fraction,
        validation_fraction=contract.validation_fraction,
    )

    # Candidate settings are selected using only train and validation. The final-test frame is
    # deliberately passed for the first time to _score_models after the policy is frozen.
    evaluation_models = _fit_models(_build_models(config), train, columns)
    policy = _select_operating_policy(
        evaluation_models,
        train,
        validation,
        columns,
        config,
    )
    metrics = _score_models(evaluation_models, validation, final_test, columns, policy)

    if verify_frozen_baseline:
        _verify_frozen_baseline(contract, dataset_path, metrics, final_test)

    rolling_folds = build_rolling_origin_folds(train, validation, contract)
    fold_reports: list[dict[str, Any]] = []
    rolling_frames: list[pd.DataFrame] = []
    rolling_predictions: list[np.ndarray] = []
    for fold in rolling_folds:
        fold_metrics, fold_prediction = _fold_report(fold, columns, policy, config)
        fold_reports.append(fold_metrics)
        rolling_frames.append(fold.score)
        rolling_predictions.append(fold_prediction)
    rolling_frame = pd.concat(rolling_frames, ignore_index=True)
    rolling_prediction = np.concatenate(rolling_predictions)

    final_test_ml = _dispatch_ml_prediction(evaluation_models, final_test, columns)
    final_test_prediction, _ = _blend_dispatch_predictions(
        final_test,
        final_test_ml,
        policy.dispatch_trend_alpha_by_horizon,
        policy.dispatch_ml_weight_by_horizon,
    )
    runtime_seconds = perf_counter() - started
    dataset_hash = _sha256(dataset_path)
    artifact_path = PROJECT_ROOT / contract.frozen_baseline["model_artifact_path"]
    report: dict[str, Any] = {
        "schema_version": 1,
        "contract": {
            "schema_version": contract.schema_version,
            "contract_id": contract.contract_id,
            "baseline_model_version": contract.baseline_model_version,
            "release_thresholds": contract.release_thresholds,
            "final_test_policy": contract.final_test_policy,
        },
        "run": {
            "run_id": (
                f"{contract.contract_id}:{contract.baseline_model_version}:"
                f"{dataset_hash[:12]}"
            ),
            "kind": "frozen_baseline_reproduction",
            "model_version": MODEL_VERSION,
            "dataset_path": (
                dataset_path.relative_to(PROJECT_ROOT).as_posix()
                if dataset_path.is_relative_to(PROJECT_ROOT)
                else dataset_path.name
            ),
            "dataset_sha256": dataset_hash,
            "rollback_artifact_path": artifact_path.relative_to(PROJECT_ROOT).as_posix(),
            "rollback_artifact_sha256": _sha256(artifact_path),
            "feature_count": len(columns),
            "feature_groups": ["current_model_feature_contract"],
            "training_config": asdict(config),
            "runtime_seconds": runtime_seconds,
        },
        "partitions": {
            "train": _partition_metadata(train),
            "validation": _partition_metadata(validation),
            "final_test": _partition_metadata(final_test),
        },
        "selection": {
            "inputs": ["train", "validation"],
            "final_test_labels_available": False,
            "operating_policy": asdict(policy),
        },
        "rolling_origin": {
            "windows": [list(window) for window in contract.rolling_origin_windows],
            "include_validation_fold": contract.include_validation_fold,
            "folds": fold_reports,
            "aggregate": _dispatch_breakdown(rolling_frame, rolling_prediction),
        },
        "final_test": {
            "sealed_during_selection": True,
            "scored_after_policy_frozen": True,
            "metrics": metrics,
            "dispatch_breakdown": _dispatch_breakdown(
                final_test,
                final_test_prediction,
            ),
        },
        "frozen_baseline_verification": {
            "enabled": verify_frozen_baseline,
            "passed": verify_frozen_baseline,
        },
    }

    dispatch_metrics = metrics["regression"]["dispatch_down_mwh"]
    registry_row = {
        "run_id": report["run"]["run_id"],
        "contract_id": contract.contract_id,
        "model_version": MODEL_VERSION,
        "dataset_sha256": dataset_hash,
        "feature_groups": json.dumps(report["run"]["feature_groups"]),
        "settings_json": json.dumps(
            {
                "training_config": asdict(config),
                "operating_policy": asdict(policy),
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        "runtime_seconds": runtime_seconds,
        "rolling_mae_mwh": report["rolling_origin"]["aggregate"]["overall"]["mae"],
        "rolling_worst_fold_mae_mwh": max(
            fold["dispatch_breakdown"]["overall"]["mae"] for fold in fold_reports
        ),
        "final_test_mae_mwh": dispatch_metrics["mae"],
        "final_test_rmse_mwh": dispatch_metrics["rmse"],
        "final_test_wape": dispatch_metrics["wape"],
        "final_test_p50_pinball_loss": metrics["uncertainty"]["p50_pinball_loss"],
        "final_test_p10_p90_coverage": metrics["uncertainty"][
            "p10_p90_empirical_coverage"
        ],
        "final_test_average_precision": metrics["classification"]["test"][
            "average_precision"
        ],
        "status": "frozen_baseline",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "benchmark_report.json"
    registry_path = output_dir / "experiment_registry.csv"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _upsert_experiment_registry(registry_path, registry_row)
    return BenchmarkResult(report, report_path, registry_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce and audit the frozen GridToEV v1.1.0 benchmark"
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--no-verify-frozen-baseline",
        action="store_true",
        help="Write a diagnostic report without comparing against frozen v1.1.0 hashes/metrics",
    )
    args = parser.parse_args()

    result = run_baseline_benchmark(
        dataset_path=args.dataset,
        contract_path=args.contract,
        output_dir=args.output_dir,
        verify_frozen_baseline=not args.no_verify_frozen_baseline,
    )
    dispatch = result.report["final_test"]["metrics"]["regression"][
        "dispatch_down_mwh"
    ]
    print(f"Benchmark contract: {result.report['contract']['contract_id']}")
    print(f"Frozen baseline verified: {result.report['frozen_baseline_verification']['passed']}")
    print(f"Final-test MAE: {dispatch['mae']:.6f} MWh")
    print(f"Final-test RMSE: {dispatch['rmse']:.6f} MWh")
    print(f"Final-test WAPE: {dispatch['wape']:.6%}")
    print(f"Report: {result.report_path}")
    print(f"Experiment registry: {result.registry_path}")


if __name__ == "__main__":
    main()
