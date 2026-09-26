"""Predeclared monthly rolling evaluation of the multi-year point candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from .constants import PROJECT_ROOT
from .history_model import (
    build_causal_history_features,
    model_feature_columns,
    source_sha256,
)


CONTRACT_PATH = PROJECT_ROOT / "config" / "multi_year_causal_experiment.v1.json"
REPORT_PATH = PROJECT_ROOT / "benchmarks" / "multi_year_causal" / "experiment_report.json"


def _score_month(
    frame: pd.DataFrame,
    month: str,
    contract: dict[str, Any],
    columns: list[str],
) -> dict[str, Any]:
    start = pd.Timestamp(f"{month}-01T00:00:00Z")
    end = start + pd.offsets.MonthBegin(1)
    # All fit labels must have matured before the first scored issue. The last
    # 60-minute score labels are also removed at the next monthly boundary.
    train = frame.loc[frame["target_timestamp_utc"].lt(start)].copy()
    score = frame.loc[
        frame["issue_timestamp_utc"].ge(start)
        & frame["issue_timestamp_utc"].lt(end)
        & frame["target_timestamp_utc"].lt(end)
    ].copy()
    if train.empty or score.empty:
        raise ValueError(f"Empty purged partition for {month}")
    by_horizon = {}
    actual_all, baseline_all, model_all = [], [], []
    for horizon in contract["forecast_horizons_minutes"]:
        fit_h = train.loc[train["forecast_horizon_minutes"].eq(horizon)]
        score_h = score.loc[score["forecast_horizon_minutes"].eq(horizon)]
        if fit_h.empty or score_h.empty:
            raise ValueError(f"Missing horizon {horizon} in {month}")
        # The contract's type is descriptive metadata, not a sklearn parameter.
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            **{key: value for key, value in contract["candidate"].items() if key != "type"},
        )
        model.fit(fit_h[columns], fit_h["dispatch_down_mwh"])
        actual = score_h["dispatch_down_mwh"].to_numpy()
        baseline = np.maximum(score_h["dispatch_down_mwh_asof"].to_numpy(), 0)
        prediction = np.maximum(model.predict(score_h[columns]), 0)
        baseline_mae = float(mean_absolute_error(actual, baseline))
        candidate_mae = float(mean_absolute_error(actual, prediction))
        by_horizon[str(horizon)] = {
            "rows": len(score_h),
            "fit_rows": len(fit_h),
            "baseline_mae_mwh": baseline_mae,
            "candidate_mae_mwh": candidate_mae,
            "mae_improvement_fraction": (baseline_mae - candidate_mae) / baseline_mae,
        }
        actual_all.append(actual)
        baseline_all.append(baseline)
        model_all.append(prediction)
    actual = np.concatenate(actual_all)
    baseline = np.concatenate(baseline_all)
    prediction = np.concatenate(model_all)
    baseline_mae = float(mean_absolute_error(actual, baseline))
    candidate_mae = float(mean_absolute_error(actual, prediction))
    return {
        "month_utc": month,
        "fit_target_max_utc": train["target_timestamp_utc"].max().isoformat(),
        "score_issue_min_utc": score["issue_timestamp_utc"].min().isoformat(),
        "score_target_max_utc": score["target_timestamp_utc"].max().isoformat(),
        "score_end_exclusive_utc": end.isoformat(),
        "label_maturity_verified": bool(
            train["target_timestamp_utc"].max() < start
            and score["target_timestamp_utc"].max() < end
        ),
        "rows": len(score),
        "baseline_mae_mwh": baseline_mae,
        "candidate_mae_mwh": candidate_mae,
        "mae_improvement_fraction": (baseline_mae - candidate_mae) / baseline_mae,
        "by_horizon": by_horizon,
    }


def evaluate_history_experiment(
    history: pd.DataFrame,
    contract: dict[str, Any],
) -> dict[str, Any]:
    frame = build_causal_history_features(
        history, observation_delay_hours=int(contract["observation_delay_hours"])
    )
    columns = model_feature_columns(frame)
    dev = [_score_month(frame, month, contract, columns) for month in contract["development_months_utc"]]
    total_rows = sum(item["rows"] for item in dev)
    baseline_mae = sum(item["baseline_mae_mwh"] * item["rows"] for item in dev) / total_rows
    candidate_mae = sum(item["candidate_mae_mwh"] * item["rows"] for item in dev) / total_rows
    improvement = (baseline_mae - candidate_mae) / baseline_mae
    gate = contract["development_gate"]
    horizon_improvements = {
        str(horizon): (
            sum(item["by_horizon"][str(horizon)]["baseline_mae_mwh"] * item["by_horizon"][str(horizon)]["rows"] for item in dev)
            - sum(item["by_horizon"][str(horizon)]["candidate_mae_mwh"] * item["by_horizon"][str(horizon)]["rows"] for item in dev)
        ) / sum(item["by_horizon"][str(horizon)]["baseline_mae_mwh"] * item["by_horizon"][str(horizon)]["rows"] for item in dev)
        for horizon in contract["forecast_horizons_minutes"]
    }
    checks = {
        "aggregate_improvement": improvement >= gate["aggregate_mae_improvement_min"],
        "fold_stability": sum(item["mae_improvement_fraction"] > 0 for item in dev) >= gate["improving_folds_min"]
        and min(item["mae_improvement_fraction"] for item in dev) >= -gate["worst_fold_regression_tolerance"],
        "both_horizons": min(horizon_improvements.values()) >= -gate["horizon_regression_tolerance"],
        "label_maturity": all(item["label_maturity_verified"] for item in dev),
    }
    passed = all(checks.values())
    # The August holdout is not even passed to a scoring model until selection
    # has completed; if development fails, it remains unscored.
    final = _score_month(frame, contract["sealed_final_month_utc"], contract, columns) if passed else None
    return {
        "experiment_id": contract["experiment_id"],
        "source_rows": len(history),
        "model_rows": len(frame),
        "feature_count": len(columns),
        "observation_delay_hours": contract["observation_delay_hours"],
        "publication_latency_verified": contract["publication_latency_verified"],
        "development": {
            "folds": dev,
            "baseline_mae_mwh": baseline_mae,
            "candidate_mae_mwh": candidate_mae,
            "aggregate_mae_improvement_fraction": improvement,
            "horizon_mae_improvement_fraction": horizon_improvements,
            "checks": checks,
            "passed": passed,
        },
        "sealed_final": {"accessed": final is not None, "score": final},
        "release": {
            "candidate_approved": False,
            "reason": "Point-only research on a different feature population; publication latency, live ingestion, v2-comparable baseline, event and interval gates remain unverified.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    source = PROJECT_ROOT / contract["source_path"]
    history = pd.read_csv(source)
    report = evaluate_history_experiment(history, contract)
    report["source_sha256"] = source_sha256(source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Development improvement: {report['development']['aggregate_mae_improvement_fraction']:.2%}")
    print(f"Development gate passed: {report['development']['passed']}")
    print(f"Sealed final accessed: {report['sealed_final']['accessed']}")
    print("Candidate approved: False")


if __name__ == "__main__":
    main()
