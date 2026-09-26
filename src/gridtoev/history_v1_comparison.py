"""Compare the multi-year research model to v1 on identical January rows.

This is an offline diagnostic, not a deployable model: retrospective EirGrid
reports do not contain verified per-row publication times.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from .benchmarking import (
    _sha256,
    _fold_report,
    build_rolling_origin_folds,
    load_benchmark_contract,
    validate_benchmark_dataset,
)
from .constants import DEFAULT_DATASET_PATH, PROJECT_ROOT
from .history_model import build_causal_history_features, model_feature_columns, source_sha256
from .training import TrainingConfig, chronological_split, feature_columns, load_dataset


CONFIG_PATH = PROJECT_ROOT / "config" / "multi_year_v1_comparison.v1.json"
REPORT_PATH = PROJECT_ROOT / "benchmarks" / "optional_v2" / "multi_year_same_rows.json"


def is_serving_eligible(
    candidate_mae: float,
    v1_mae: float,
    *,
    publication_verified: bool,
    final_scored: bool,
) -> bool:
    """Lower development MAE cannot waive publication or final-test gates."""

    return bool(publication_verified and final_scored and candidate_mae < v1_mae)


def align_history_to_v1(
    v1_rows: pd.DataFrame,
    history_rows: pd.DataFrame,
    *,
    delay_hours: int,
) -> pd.DataFrame:
    """Require exact issue/target/horizon/label identity and a delayed feature row."""

    if delay_hours <= 0:
        raise ValueError("A positive observation delay is required")
    keys = ["issue_timestamp_utc", "target_timestamp_utc", "forecast_horizon_minutes"]
    left = v1_rows[keys + ["dispatch_down_mwh"]].copy()
    right = history_rows.copy()
    for frame in (left, right):
        for name in (*keys[:2],):
            frame[name] = pd.to_datetime(frame[name], utc=True, errors="raise")
    for frame in (left, right):
        if frame.duplicated(keys).any():
            raise ValueError("Duplicate issue/horizon keys prevent a fair comparison")
    left["_v1_order"] = np.arange(len(left))
    right = right.rename(columns={"dispatch_down_mwh": "_history_label"})
    aligned = left.merge(right, on=keys, how="left", validate="one_to_one", indicator=True)
    if aligned["_merge"].ne("both").any():
        raise ValueError("Historical features do not cover every v1 comparison row")
    if not np.allclose(
        aligned["dispatch_down_mwh"], aligned["_history_label"], atol=1e-6, rtol=0
    ):
        raise ValueError("Historical and v1 label values disagree")
    aligned["feature_timestamp_utc"] = pd.to_datetime(
        aligned["feature_timestamp_utc"], utc=True, errors="raise"
    )
    cutoff = aligned["issue_timestamp_utc"] - pd.Timedelta(hours=delay_hours)
    if aligned["feature_timestamp_utc"].gt(cutoff).any():
        raise ValueError("Historical feature timestamp crosses observation cutoff")
    expected_target = aligned["issue_timestamp_utc"] + pd.to_timedelta(
        aligned["forecast_horizon_minutes"], unit="m"
    )
    if aligned["target_timestamp_utc"].ne(expected_target).any():
        raise ValueError("Historical target timestamp does not match the horizon")
    return aligned.sort_values("_v1_order").drop(columns=["_v1_order", "_merge", "_history_label"])


def evaluate(
    *,
    config_path: Path = CONFIG_PATH,
    output_path: Path = REPORT_PATH,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset = PROJECT_ROOT / config["v1_dataset_path"]
    history_path = PROJECT_ROOT / config["history_path"]
    contract = load_benchmark_contract(PROJECT_ROOT / config["benchmark_contract_path"])
    if contract.label_maturity_policy != "target_before_score_issue":
        raise ValueError("The v1 comparison requires the purged benchmark")
    validate_benchmark_dataset(dataset)
    if _sha256(dataset) != contract.frozen_baseline["dataset_sha256"]:
        raise ValueError("v1 dataset differs from the frozen benchmark")
    if source_sha256(history_path) != config["history_sha256"]:
        raise ValueError("Multi-year source differs from the predeclared history")
    v1 = load_dataset(dataset)
    train, validation, final = chronological_split(
        v1, contract.train_fraction, contract.validation_fraction
    )
    folds = build_rolling_origin_folds(
        train, validation, contract, final_test_start_utc=final.issue_timestamp_utc.min()
    )
    history = pd.read_csv(history_path)
    historical_rows = build_causal_history_features(
        history, observation_delay_hours=int(config["observation_delay_hours"])
    )
    columns = model_feature_columns(historical_rows)
    training_end = pd.Timestamp(config["historical_train_target_before_utc"])
    fit = historical_rows.loc[historical_rows.target_timestamp_utc.lt(training_end)]
    if fit.empty or fit.target_timestamp_utc.max() >= training_end:
        raise ValueError("No label-mature historical training rows")
    models: dict[int, HistGradientBoostingRegressor] = {}
    for horizon in config["forecast_horizons_minutes"]:
        subset = fit.loc[fit.forecast_horizon_minutes.eq(horizon)]
        if subset.empty:
            raise ValueError(f"No historical training rows for horizon {horizon}")
        models[horizon] = HistGradientBoostingRegressor(
            **{key: value for key, value in config["candidate"].items() if key != "type"}
        ).fit(subset[columns], subset.dispatch_down_mwh)

    fold_reports: list[dict[str, Any]] = []
    totals: list[tuple[int, float, float]] = []
    for fold in folds:
        _, v1_prediction = _fold_report(
            fold, feature_columns(v1), None,
            TrainingConfig(
                train_fraction=contract.train_fraction,
                validation_fraction=contract.validation_fraction,
            ),
            contract,
        )
        aligned = align_history_to_v1(
            fold.score, historical_rows,
            delay_hours=int(config["observation_delay_hours"]),
        )
        candidate = np.empty(len(aligned))
        for horizon, model in models.items():
            mask = aligned.forecast_horizon_minutes.eq(horizon).to_numpy()
            candidate[mask] = np.maximum(model.predict(aligned.loc[mask, columns]), 0)
        truth = aligned.dispatch_down_mwh.to_numpy()
        v1_mae = float(mean_absolute_error(truth, v1_prediction))
        candidate_mae = float(mean_absolute_error(truth, candidate))
        fold_reports.append({
            "fold": fold.name,
            "rows": len(aligned),
            "v1_mae_mwh": v1_mae,
            "candidate_mae_mwh": candidate_mae,
        })
        totals.append((len(aligned), v1_mae, candidate_mae))
    row_count = sum(row[0] for row in totals)
    v1_mae = sum(count * mae for count, mae, _ in totals) / row_count
    candidate_mae = sum(count * mae for count, _, mae in totals) / row_count
    if not np.isclose(
        v1_mae, float(contract.frozen_baseline["rolling_mae_mwh"]), atol=1e-9, rtol=0
    ):
        raise ValueError("v1 comparison does not reproduce the frozen purged baseline")
    publication_verified = bool(config["publication_latency_verified"])
    report: dict[str, Any] = {
        "experiment_id": config["experiment_id"],
        "history_sha256": source_sha256(history_path),
        "historical_source_rows": len(history),
        "historical_training_rows": len(fit),
        "historical_feature_count": len(columns),
        "historical_train_target_max_utc": fit.target_timestamp_utc.max().isoformat(),
        "observation_delay_hours": config["observation_delay_hours"],
        "publication_latency_verified": publication_verified,
        "development": {
            "rows": row_count,
            "v1_mae_mwh": v1_mae,
            "candidate_mae_mwh": candidate_mae,
            "mae_improvement_fraction": (v1_mae - candidate_mae) / v1_mae,
            "folds": fold_reports,
        },
        "sealed_final_scored": False,
        "serving_eligible": is_serving_eligible(
            candidate_mae, v1_mae,
            publication_verified=publication_verified,
            final_scored=False,
        ),
        "artifact_created": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
