"""Leakage-safe interval calibration and component reconciliation research.

Selection sees only issue #8's development out-of-fold predictions. The sealed
final period is loaded by the runner only after the policy has been frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_pinball_loss

from .benchmarking import load_benchmark_contract
from .constants import DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH, PROJECT_ROOT
from .training import (
    _blend_dispatch_predictions,
    _dispatch_ml_prediction,
    chronological_split,
    feature_columns,
    load_dataset,
)


DEFAULT_OOF_PATH = PROJECT_ROOT / "benchmarks" / "horizon_models" / "development_oof_predictions.csv.gz"
DEFAULT_HORIZON_REPORT = PROJECT_ROOT / "benchmarks" / "horizon_models" / "benchmark_report.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "benchmarks" / "calibrated_intervals"
CALIBRATION_METHODS = (
    "global_symmetric",
    "horizon_symmetric",
    "horizon_asymmetric",
    "horizon_recent_symmetric",
)


@dataclass(frozen=True)
class IntervalPolicy:
    method: str
    central_forecast: str
    nominal_coverage: float
    offsets_by_horizon: dict[str, dict[str, float]]
    calibration_rows: int
    calibration_fold_names: list[str]
    selected_from: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_oof_predictions(frame: pd.DataFrame) -> None:
    required = {
        "issue_timestamp_utc",
        "forecast_horizon_minutes",
        "dispatch_down_mwh",
        "v1.1.0",
        "fold",
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"OOF predictions are missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("OOF predictions are empty")
    if frame["fold"].astype(str).str.contains("final.test", case=False, regex=True).any():
        raise ValueError("OOF selection must never include final-test rows")
    if frame.duplicated(["issue_timestamp_utc", "forecast_horizon_minutes"]).any():
        raise ValueError("OOF predictions contain duplicate issue/horizon keys")
    if frame[list(required - {"issue_timestamp_utc", "fold"})].isna().any().any():
        raise ValueError("OOF predictions contain missing numeric values")
    if not np.isfinite(frame[["dispatch_down_mwh", "v1.1.0"]].to_numpy(dtype=float)).all():
        raise ValueError("OOF predictions contain non-finite values")


def _higher_quantile(values: np.ndarray, probability: float) -> float:
    """Finite-sample conformal order statistic, clamped to observed support."""
    ordered = np.sort(np.asarray(values, dtype=float))
    if len(ordered) == 0:
        raise ValueError("Calibration requires at least one residual")
    index = min(len(ordered) - 1, max(0, int(np.ceil((len(ordered) + 1) * probability)) - 1))
    return float(ordered[index])


def fit_interval_policy(
    oof: pd.DataFrame,
    method: str,
    *,
    nominal_coverage: float = 0.80,
) -> IntervalPolicy:
    validate_oof_predictions(oof)
    if method not in CALIBRATION_METHODS:
        raise ValueError(f"Unsupported calibration method: {method}")
    if not 0 < nominal_coverage < 1:
        raise ValueError("Nominal coverage must lie strictly between zero and one")
    frame = oof
    if method == "horizon_recent_symmetric":
        # The caller supplies folds in chronological order. Calibrate with the
        # most recent scored fold when the method explicitly requests recency.
        latest_fold = str(frame["fold"].iloc[-1])
        frame = frame.loc[frame["fold"].eq(latest_fold)].copy()
    residual = frame["dispatch_down_mwh"].to_numpy(dtype=float) - frame["v1.1.0"].to_numpy(dtype=float)
    horizons = frame["forecast_horizon_minutes"].to_numpy(dtype=int)
    offsets: dict[str, dict[str, float]] = {}
    for horizon in sorted(np.unique(horizons)):
        local = residual if method == "global_symmetric" else residual[horizons == horizon]
        if method == "horizon_asymmetric":
            tail = (1.0 - nominal_coverage) / 2.0
            lower = min(float(np.quantile(local, tail, method="lower")), 0.0)
            upper = max(_higher_quantile(local, 1.0 - tail), 0.0)
        else:
            radius = _higher_quantile(np.abs(local), nominal_coverage)
            lower, upper = -radius, radius
        offsets[str(horizon)] = {"lower": lower, "upper": upper}
    return IntervalPolicy(
        method=method,
        central_forecast="v1.1.0 dispatch-down trend",
        nominal_coverage=nominal_coverage,
        offsets_by_horizon=offsets,
        calibration_rows=len(frame),
        calibration_fold_names=list(dict.fromkeys(frame["fold"].astype(str))),
        selected_from="development out-of-fold predictions only",
    )


def apply_interval_policy(
    horizons: np.ndarray,
    central_predictions: np.ndarray,
    policy: IntervalPolicy,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    horizon_values = np.asarray(horizons, dtype=int)
    p50 = np.maximum(np.asarray(central_predictions, dtype=float), 0.0)
    if len(horizon_values) != len(p50):
        raise ValueError("Horizons and central predictions must have equal lengths")
    lower = np.empty(len(p50), dtype=float)
    upper = np.empty(len(p50), dtype=float)
    for horizon in np.unique(horizon_values):
        if str(horizon) not in policy.offsets_by_horizon:
            raise ValueError(f"No calibration for {horizon}-minute horizon")
        mask = horizon_values == horizon
        offsets = policy.offsets_by_horizon[str(horizon)]
        lower[mask] = np.maximum(p50[mask] + min(offsets["lower"], 0.0), 0.0)
        upper[mask] = p50[mask] + max(offsets["upper"], 0.0)
    return lower, p50, upper


def _interval_metrics(truth: np.ndarray, lower: np.ndarray, central: np.ndarray, upper: np.ndarray) -> dict[str, float]:
    inside = (truth >= lower) & (truth <= upper)
    return {
        "rows": int(len(truth)),
        "coverage": float(inside.mean()),
        "average_width_mwh": float(np.mean(upper - lower)),
        "median_width_mwh": float(np.median(upper - lower)),
        "p10_pinball_loss": float(mean_pinball_loss(truth, lower, alpha=0.10)),
        "p50_pinball_loss": float(mean_pinball_loss(truth, central, alpha=0.50)),
        "p90_pinball_loss": float(mean_pinball_loss(truth, upper, alpha=0.90)),
    }


def _ordered_folds(oof: pd.DataFrame) -> list[str]:
    order = (
        oof.assign(issue=pd.to_datetime(oof["issue_timestamp_utc"], utc=True))
        .groupby("fold", sort=False)["issue"]
        .min()
        .sort_values()
    )
    return [str(name) for name in order.index]


def select_interval_policy(oof: pd.DataFrame) -> tuple[IntervalPolicy, pd.DataFrame]:
    """Use forward fold comparisons, then fit the selected method on development OOF."""
    validate_oof_predictions(oof)
    order = _ordered_folds(oof)
    if len(order) < 3:
        raise ValueError("At least three chronological OOF folds are required")
    rows: list[dict[str, Any]] = []
    for method in CALIBRATION_METHODS:
        for index in range(1, len(order)):
            calibration = oof.loc[oof["fold"].isin(order[:index])].copy()
            scoring = oof.loc[oof["fold"].eq(order[index])].copy()
            policy = fit_interval_policy(calibration, method)
            low, middle, high = apply_interval_policy(
                scoring["forecast_horizon_minutes"].to_numpy(),
                scoring["v1.1.0"].to_numpy(),
                policy,
            )
            truth = scoring["dispatch_down_mwh"].to_numpy(dtype=float)
            for horizon in sorted(scoring["forecast_horizon_minutes"].unique()):
                mask = scoring["forecast_horizon_minutes"].eq(horizon).to_numpy()
                rows.append(
                    {
                        "method": method,
                        "score_fold": order[index],
                        "horizon_minutes": int(horizon),
                        "calibration_rows": policy.calibration_rows,
                        **_interval_metrics(
                            truth[mask], low[mask], middle[mask], high[mask]
                        ),
                    }
                )
    comparison = pd.DataFrame(rows)
    ranking: list[tuple[float, float, float, str]] = []
    for method in CALIBRATION_METHODS:
        subset = comparison.loc[comparison["method"].eq(method)]
        coverage = subset["coverage"].to_numpy(dtype=float)
        violations = np.maximum(0.78 - coverage, 0) + np.maximum(coverage - 0.90, 0)
        ranking.append(
            (
                float(violations.mean()),
                float(np.abs(coverage - 0.80).mean()),
                float(subset["average_width_mwh"].mean()),
                method,
            )
        )
    chosen = min(ranking)[-1]
    policy = fit_interval_policy(oof, chosen)
    return policy, comparison


def reconcile_components(
    total: np.ndarray,
    raw_curtailment: np.ndarray,
    raw_constraint: np.ndarray,
    *,
    default_curtailment_share: float,
) -> tuple[np.ndarray, np.ndarray]:
    total = np.maximum(np.asarray(total, dtype=float), 0.0)
    curtailment_raw = np.maximum(np.asarray(raw_curtailment, dtype=float), 0.0)
    constraint_raw = np.maximum(np.asarray(raw_constraint, dtype=float), 0.0)
    if len(total) != len(curtailment_raw) or len(total) != len(constraint_raw):
        raise ValueError("Total and component arrays must have equal lengths")
    if not 0 <= default_curtailment_share <= 1:
        raise ValueError("Default curtailment share must be between zero and one")
    raw_sum = curtailment_raw + constraint_raw
    ratio = np.divide(
        curtailment_raw,
        raw_sum,
        out=np.full(len(total), default_curtailment_share, dtype=float),
        where=raw_sum > 0,
    )
    curtailment = total * ratio
    constraint = total - curtailment
    return curtailment, constraint


def assess_component_refit(
    development: pd.DataFrame,
    *,
    minimum_history_days: int = 90,
    minimum_positive_rows: int = 100,
    minimum_positive_days: int = 20,
) -> dict[str, Any]:
    issue = pd.to_datetime(development["issue_timestamp_utc"], utc=True)
    span_days = float((issue.max() - issue.min()).total_seconds() / 86400)
    positive_rows = {}
    positive_days = {}
    for component in ("curtailment_mwh", "constraint_mwh"):
        positive = development[component].gt(0)
        positive_rows[component] = int(positive.sum())
        positive_days[component] = int(issue.loc[positive].dt.floor("D").nunique())
    checks = {
        "history_days": span_days >= minimum_history_days,
        "positive_rows": min(positive_rows.values()) >= minimum_positive_rows,
        "positive_days": min(positive_days.values()) >= minimum_positive_days,
    }
    return {
        "eligible": all(checks.values()),
        "checks": checks,
        "failed_checks": [key for key, passed in checks.items() if not passed],
        "history_span_days": span_days,
        "positive_rows": positive_rows,
        "positive_days": positive_days,
        "requirements": {
            "minimum_history_days": minimum_history_days,
            "minimum_positive_rows_per_component": minimum_positive_rows,
            "minimum_positive_days_per_component": minimum_positive_days,
        },
    }


def _baseline_bundle_intervals(bundle: dict[str, Any], frame: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    models = bundle["models"]
    latest = frame["dispatch_down_mwh_latest_observed"].to_numpy(dtype=float)
    raw = np.column_stack(
        [
            np.maximum(
                latest + models[f"dispatch_down_quantile_{name}"].predict(frame[columns]),
                0.0,
            )
            for name in ("p10", "p50", "p90")
        ]
    )
    raw.sort(axis=1)
    adjustment = float(bundle["metadata"]["prediction_interval_adjustment_mwh"])
    return np.maximum(raw[:, 0] - adjustment, 0.0), raw[:, 1], raw[:, 2] + adjustment


def run_calibration_benchmark(
    *,
    oof_path: Path | str = DEFAULT_OOF_PATH,
    horizon_report_path: Path | str = DEFAULT_HORIZON_REPORT,
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
    model_path: Path | str = DEFAULT_MODEL_PATH,
    contract_path: Path | str = PROJECT_ROOT / "config" / "benchmark_contract.v1.json",
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    oof_path = Path(oof_path)
    dataset_path = Path(dataset_path)
    contract = load_benchmark_contract(contract_path)
    horizon_report = json.loads(Path(horizon_report_path).read_text(encoding="utf-8"))
    if _sha256(dataset_path) != horizon_report["dataset"]["sha256"]:
        raise ValueError("Dataset hash differs from issue #8 OOF benchmark")
    if horizon_report["contract"]["contract_id"] != contract.contract_id:
        raise ValueError("OOF benchmark uses a different fold contract")
    oof = pd.read_csv(oof_path)
    validate_oof_predictions(oof)
    final_min = pd.Timestamp(contract.frozen_baseline["final_test_period"]["issue_timestamp_min_utc"])
    if pd.to_datetime(oof["issue_timestamp_utc"], utc=True).max() >= final_min:
        raise ValueError("OOF calibration rows overlap the sealed final-test period")

    # The selection interface accepts OOF predictions only. Load labelled final-test
    # data only after the method and offsets have been fixed.
    policy, comparison = select_interval_policy(oof)
    frozen_policy = asdict(policy)

    data = load_dataset(dataset_path)
    columns = feature_columns(data)
    train, validation, final_test = chronological_split(
        data,
        train_fraction=contract.train_fraction,
        validation_fraction=contract.validation_fraction,
    )
    if final_test["issue_timestamp_utc"].min() != final_min:
        raise ValueError("Final-test period differs from the frozen contract")
    bundle = joblib.load(model_path)
    if _sha256(Path(model_path)) != contract.frozen_baseline["model_artifact_sha256"]:
        raise ValueError("Production bundle hash differs from the frozen v1.1.0 artifact")
    if list(bundle["feature_columns"]) != columns:
        raise ValueError("Production bundle feature contract differs from the dataset")

    metadata = bundle["metadata"]
    ml = _dispatch_ml_prediction(bundle["models"], final_test, columns)
    central, _ = _blend_dispatch_predictions(
        final_test,
        ml,
        metadata["dispatch_trend_alpha_by_horizon"],
        metadata["dispatch_regression_ml_weight_by_horizon"],
    )
    horizons = final_test["forecast_horizon_minutes"].to_numpy(dtype=int)
    p10, p50, p90 = apply_interval_policy(horizons, central, policy)
    old_p10, old_p50, old_p90 = _baseline_bundle_intervals(bundle, final_test, columns)
    truth = final_test["dispatch_down_mwh"].to_numpy(dtype=float)
    candidate_metrics = _interval_metrics(truth, p10, p50, p90)
    baseline_metrics = _interval_metrics(truth, old_p10, old_p50, old_p90)
    by_horizon = {}
    for horizon in sorted(np.unique(horizons)):
        mask = horizons == horizon
        by_horizon[str(horizon)] = {
            "candidate": _interval_metrics(truth[mask], p10[mask], p50[mask], p90[mask]),
            "deployed_v1.1.0": _interval_metrics(
                truth[mask], old_p10[mask], old_p50[mask], old_p90[mask]
            ),
        }

    raw_curtailment = bundle["models"]["curtailment_regressor"].predict(final_test[columns])
    raw_constraint = bundle["models"]["constraint_regressor"].predict(final_test[columns])
    curtailment, constraint = reconcile_components(
        central,
        raw_curtailment,
        raw_constraint,
        default_curtailment_share=float(metadata["default_component_shares"]["curtailment"]),
    )
    refit = assess_component_refit(pd.concat([train, validation], ignore_index=True))
    reconciliation_error = float(np.max(np.abs(curtailment + constraint - central)))
    quantiles_ordered = bool(
        np.all(p10 >= 0) and np.all(p10 <= p50) and np.all(p50 <= p90)
    )
    width_ratio = (
        candidate_metrics["average_width_mwh"] / baseline_metrics["average_width_mwh"]
        if baseline_metrics["average_width_mwh"] > 0 else float("inf")
    )
    checks = {
        "p50_pinball": candidate_metrics["p50_pinball_loss"]
        <= contract.release_thresholds["final_test_p50_pinball_loss_max"],
        "coverage_min": candidate_metrics["coverage"]
        >= contract.release_thresholds["final_test_p10_p90_coverage_min"],
        "coverage_max": candidate_metrics["coverage"]
        <= contract.release_thresholds["final_test_p10_p90_coverage_max"],
        "width_not_excessive": width_ratio <= 1.25,
        "quantiles_ordered_nonnegative": quantiles_ordered,
        "components_reconciled": reconciliation_error <= 1e-9,
    }
    report = {
        "schema_version": 1,
        "issue": 9,
        "contract_id": contract.contract_id,
        "dataset_sha256": _sha256(dataset_path),
        "oof_sha256": _sha256(oof_path),
        "production_bundle_sha256": _sha256(Path(model_path)),
        "selection": {
            "selected_before_final_test_scoring": True,
            "policy": frozen_policy,
            "development_folds": _ordered_folds(oof),
            "final_test_labels_used": False,
        },
        "final_test": {
            "accessed_after_selection": True,
            "candidate": candidate_metrics,
            "deployed_v1.1.0": baseline_metrics,
            "by_horizon": by_horizon,
            "average_width_ratio_to_deployed_v1.1.0": width_ratio,
        },
        "components": {
            "refit_eligibility": refit,
            "refitted": False,
            "reconciliation_max_absolute_error_mwh": reconciliation_error,
            "curtailment_mae_mwh": float(mean_absolute_error(final_test["curtailment_mwh"], curtailment)),
            "constraint_mae_mwh": float(mean_absolute_error(final_test["constraint_mwh"], constraint)),
        },
        "release_checks": checks,
        "release_candidate_eligible": all(checks.values()),
        "production_decision": "candidate_for_issue_10_integration" if all(checks.values()) else "retain_v1.1.0_intervals",
        "production_bundle_modified": False,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "calibration_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "interval_policy.json").write_text(
        json.dumps(frozen_policy, indent=2, sort_keys=True), encoding="utf-8"
    )
    comparison.to_csv(output_dir / "development_interval_comparison.csv", index=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate dispatch-down intervals from development OOF predictions")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    report = run_calibration_benchmark(output_dir=args.output_dir)
    print(f"Selected method: {report['selection']['policy']['method']}")
    print(f"P50 pinball: {report['final_test']['candidate']['p50_pinball_loss']:.4f}")
    print(f"Coverage: {report['final_test']['candidate']['coverage']:.2%}")
    print(f"Average width: {report['final_test']['candidate']['average_width_mwh']:.2f} MWh")
    print(f"Production decision: {report['production_decision']}")


if __name__ == "__main__":
    main()
