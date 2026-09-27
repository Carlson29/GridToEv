"""Optional UTC-day curtailment-risk and energy model, separate from v1."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, brier_score_loss, mean_absolute_error, roc_auc_score

from .constants import PROJECT_ROOT
from .daily_curtailment import (
    DEFAULT_DAILY_DATASET,
    FORECAST_MODEL,
    REGIONS,
    build_forecast_features,
    fetch_live_forecast,
    fetch_previous_runs,
)


DEFAULT_ARTIFACT = PROJECT_ROOT / "models" / "v2" / "daily_curtailment_bundle.joblib"
DEFAULT_REPORT = PROJECT_ROOT / "benchmarks" / "daily_curtailment_v2" / "evaluation.json"
EARLIEST_TARGET_DATE = date(2024, 4, 1)
NON_FEATURES = {
    "issue_timestamp_utc",
    "forecast_max_available_at_utc",
    "curtailment_mwh",
    "curtailment_event",
}
REGIONAL_METRICS = (
    "wind_mean_kmh",
    "wind_max_kmh",
    "wind_p75_kmh",
    "solar_sum_wm2h",
    "temperature_mean_c",
)
ALLOWED_FEATURES = {
    *(f"{region}_{metric}" for region in REGIONS for metric in REGIONAL_METRICS),
    "wind_regional_spread_kmh",
    "calendar_day_sin",
    "calendar_day_cos",
    "calendar_weekend",
}


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Allow only explicitly forecast-derived and calendar columns."""

    columns = [column for column in frame.columns if column in ALLOWED_FEATURES]
    if not columns:
        raise ValueError("No forecast feature columns")
    return columns


def chronological_partitions(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(frame["issue_timestamp_utc"], utc=True)
    return (
        frame.loc[dates.lt("2025-01-01T00:00:00Z")].copy(),
        frame.loc[dates.ge("2025-01-01T00:00:00Z") & dates.lt("2026-01-01T00:00:00Z")].copy(),
        frame.loc[dates.ge("2026-01-01T00:00:00Z")].copy(),
    )


def _validate_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    required = NON_FEATURES | ALLOWED_FEATURES
    if required - set(data):
        raise ValueError(f"Daily dataset missing {sorted(required - set(data))}")
    data["issue_timestamp_utc"] = pd.to_datetime(data["issue_timestamp_utc"], utc=True)
    data["forecast_max_available_at_utc"] = pd.to_datetime(data["forecast_max_available_at_utc"], utc=True)
    if data["issue_timestamp_utc"].duplicated().any():
        raise ValueError("Duplicate prediction day")
    if data["issue_timestamp_utc"].dt.hour.ne(0).any():
        raise ValueError("Prediction issues must be at 00:00 UTC")
    if (data["forecast_max_available_at_utc"] > data["issue_timestamp_utc"]).any():
        raise ValueError("Forecast input became available after prediction issue")
    columns = feature_columns(data)
    if data[columns].isna().any().any() or not np.isfinite(data[columns].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite forecast features")
    if data["curtailment_mwh"].isna().any() or data["curtailment_mwh"].lt(0).any():
        raise ValueError("Invalid curtailment label")
    if not data["curtailment_event"].eq(data["curtailment_mwh"].gt(0).astype(int)).all():
        raise ValueError("Curtailment event and energy labels disagree")
    return data.sort_values("issue_timestamp_utc").reset_index(drop=True)


def _classifier() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=120, max_leaf_nodes=10, min_samples_leaf=12,
        learning_rate=0.045, l2_regularization=3.0, random_state=42,
    )


def _regressors() -> dict[str, object]:
    return {
        "direct_median": HistGradientBoostingRegressor(
            loss="absolute_error", max_iter=150, max_leaf_nodes=10,
            min_samples_leaf=12, learning_rate=0.05, l2_regularization=3.0,
            random_state=42,
        ),
        "two_stage": ExtraTreesRegressor(
            n_estimators=240, min_samples_leaf=4, max_features=0.85,
            n_jobs=-1, random_state=42,
        ),
    }


def _metrics(frame: pd.DataFrame, probability: np.ndarray, amount: np.ndarray) -> dict:
    truth_event = frame["curtailment_event"].to_numpy(dtype=int)
    truth_amount = frame["curtailment_mwh"].to_numpy(dtype=float)
    positive = truth_event.astype(bool)
    return {
        "rows": len(frame),
        "event_rate": float(truth_event.mean()),
        "event_average_precision": float(average_precision_score(truth_event, probability)),
        "event_brier_score": float(brier_score_loss(truth_event, probability)),
        "event_roc_auc": float(roc_auc_score(truth_event, probability)) if len(np.unique(truth_event)) == 2 else None,
        "daily_mae_mwh": float(mean_absolute_error(truth_amount, amount)),
        "positive_day_mae_mwh": float(mean_absolute_error(truth_amount[positive], amount[positive])) if positive.any() else None,
        "predicted_positive_fraction": float(np.mean(amount > 0)),
    }


def _predict_amount(method: str, regressor: object, features: pd.DataFrame, probability: np.ndarray) -> np.ndarray:
    positive_amount = np.maximum(regressor.predict(features), 0)
    if method == "two_stage":
        return probability * positive_amount
    return positive_amount


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_daily_model(
    dataset_path: Path = DEFAULT_DAILY_DATASET,
    artifact_path: Path = DEFAULT_ARTIFACT,
    report_path: Path = DEFAULT_REPORT,
) -> dict:
    """Select on 2025, score 2026 once, and save that evaluated fitted model."""

    data = _validate_dataset(pd.read_csv(dataset_path))
    train, validation, test = chronological_partitions(data)
    if min(len(train), len(validation), len(test)) < 30:
        raise ValueError("Need at least 30 complete days in each time partition")
    if train["curtailment_event"].nunique() < 2:
        raise ValueError("Training requires event and non-event days")
    columns = feature_columns(data)
    event_model = _classifier().fit(train[columns], train["curtailment_event"])
    val_probability = event_model.predict_proba(validation[columns])[:, 1]
    candidates = _regressors()
    validation_scores = {}
    for method, regressor in candidates.items():
        fit = train.loc[train["curtailment_event"].eq(1)] if method == "two_stage" else train
        regressor.fit(fit[columns], fit["curtailment_mwh"])
        amount = _predict_amount(method, regressor, validation[columns], val_probability)
        validation_scores[method] = float(mean_absolute_error(validation["curtailment_mwh"], amount))
    selected = min(validation_scores, key=lambda key: (validation_scores[key], key))
    development = pd.concat([train, validation], ignore_index=True)
    event_model = _classifier().fit(development[columns], development["curtailment_event"])
    amount_model = _regressors()[selected]
    fit = development.loc[development["curtailment_event"].eq(1)] if selected == "two_stage" else development
    amount_model.fit(fit[columns], fit["curtailment_mwh"])
    test_probability = event_model.predict_proba(test[columns])[:, 1]
    test_amount = _predict_amount(selected, amount_model, test[columns], test_probability)
    prevalence = float(development["curtailment_event"].mean())
    zero = np.zeros(len(test))
    month_medians = development.groupby(development["issue_timestamp_utc"].dt.month)["curtailment_mwh"].median()
    seasonal_amount = test["issue_timestamp_utc"].dt.month.map(month_medians).fillna(0).to_numpy(dtype=float)
    by_quarter = {}
    quarters = test["issue_timestamp_utc"].dt.year.astype(str) + "Q" + test["issue_timestamp_utc"].dt.quarter.astype(str)
    for quarter, subset in test.groupby(quarters):
        positions = test.index.get_indexer(subset.index)
        by_quarter[str(quarter)] = {
            "rows": len(subset),
            "model_mae_mwh": float(mean_absolute_error(subset["curtailment_mwh"], test_amount[positions])),
            "zero_mae_mwh": float(mean_absolute_error(subset["curtailment_mwh"], np.zeros(len(subset)))),
        }
    report = {
        "model_version": "2.0.0-daily-experimental",
        "task": "UTC-day curtailment event and energy, issued at 00:00 UTC",
        "not_comparable_to_v1_mae": "v1 predicts half-hour dispatch-down, not daily curtailment",
        "source": f"EirGrid curtailment labels; Open-Meteo Previous Runs {FORECAST_MODEL} previous_day1 archived forecast fields",
        "forecast_lead_rule": "Each hourly forecast is for target hour minus 24 hours; latest daily forecast D-1 23:00 UTC",
        "dataset_sha256": _sha256(dataset_path),
        "date_ranges": {
            "train": [train["issue_timestamp_utc"].min().isoformat(), train["issue_timestamp_utc"].max().isoformat()],
            "validation": [validation["issue_timestamp_utc"].min().isoformat(), validation["issue_timestamp_utc"].max().isoformat()],
            "test": [test["issue_timestamp_utc"].min().isoformat(), test["issue_timestamp_utc"].max().isoformat()],
        },
        "rows": {"train": len(train), "validation": len(validation), "test": len(test)},
        "selected_amount_method": selected,
        "validation_amount_mae_mwh": validation_scores,
        "test": _metrics(test, test_probability, test_amount),
        "test_zero_amount_baseline": _metrics(test, np.full(len(test), prevalence), zero),
        "test_monthly_median_baseline_mae_mwh": float(mean_absolute_error(test["curtailment_mwh"], seasonal_amount)),
        "test_by_quarter": by_quarter,
        "source_caveat": "Previous Runs has a lead-time convention, not immutable per-hour publication timestamps; recheck live vintages before production promotion.",
    }
    artifact = {
        "metadata": {
            "model_version": report["model_version"],
            "task": report["task"],
            "dataset_sha256": report["dataset_sha256"],
            "feature_columns": columns,
            "selected_amount_method": selected,
            "trained_through_utc": development["issue_timestamp_utc"].max().isoformat(),
            "test_daily_mae_mwh": report["test"]["daily_mae_mwh"],
            "test_zero_baseline_mae_mwh": report["test_zero_amount_baseline"]["daily_mae_mwh"],
            "experimental": True,
        },
        "event_model": event_model,
        "amount_model": amount_model,
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_path, compress=3)
    report["artifact_sha256"] = _sha256(artifact_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


class DailyCurtailmentService:
    def __init__(
        self,
        artifact_path: Path = DEFAULT_ARTIFACT,
        report_path: Path = DEFAULT_REPORT,
    ) -> None:
        self.artifact_path = Path(artifact_path)
        self.report_path = Path(report_path)
        self.bundle: dict | None = None

    def load(self) -> None:
        report = json.loads(self.report_path.read_text(encoding="utf-8"))
        if _sha256(self.artifact_path) != report["artifact_sha256"]:
            raise ValueError("Daily model artifact checksum does not match its evaluation report")
        bundle = joblib.load(self.artifact_path)
        if bundle["metadata"]["task"] != "UTC-day curtailment event and energy, issued at 00:00 UTC":
            raise ValueError("Not a compatible daily curtailment bundle")
        if bundle["metadata"]["dataset_sha256"] != report["dataset_sha256"]:
            raise ValueError("Daily model training dataset hash does not match its evaluation report")
        self.bundle = bundle

    def model_info(self) -> dict:
        if self.bundle is None:
            self.load()
        assert self.bundle is not None
        report = json.loads(self.report_path.read_text(encoding="utf-8"))
        info = dict(self.bundle["metadata"])
        info["model_id"] = "v2"
        info["target_description"] = "Whether any curtailment occurs and total curtailment MWh over one UTC day"
        info["metric_guide"] = {
            "daily_mae_mwh": "Average absolute miss in total curtailment energy per complete UTC day, MWh; lower is better.",
            "event_average_precision": "How well event probabilities rank curtailment days; higher is better and compare with event rate.",
            "event_brier_score": "Average squared probability error for the curtailment event; lower is better.",
            "event_roc_auc": "Ranking of curtailment versus non-curtailment days; higher is better.",
            "positive_day_mae_mwh": "Average amount error only on days when curtailment actually occurred, MWh; lower is better.",
        }
        info["prediction_components"] = {
            "event_model": {
                "estimator": type(self.bundle["event_model"]).__name__,
                "hyperparameters": {
                    key: self.bundle["event_model"].get_params(deep=False)[key]
                    for key in ("max_iter", "max_leaf_nodes", "min_samples_leaf", "learning_rate", "l2_regularization", "random_state")
                },
                "role": "Probability that curtailment occurs during the day",
                "test_metrics": {
                    key: report["test"][key]
                    for key in ("event_average_precision", "event_brier_score", "event_roc_auc")
                },
            },
            "amount_model": {
                "estimator": type(self.bundle["amount_model"]).__name__,
                "hyperparameters": {
                    key: value for key, value in self.bundle["amount_model"].get_params(deep=False).items()
                    if key in ("n_estimators", "min_samples_leaf", "max_features", "n_jobs", "random_state", "loss", "max_iter", "max_leaf_nodes", "learning_rate", "l2_regularization")
                },
                "role": "Positive-day curtailment MWh; the two-stage serving estimate multiplies it by event probability",
                "selected_method": info["selected_amount_method"],
                "separate_test_score_available": False,
            },
        }
        info["evaluation"] = {
            "train_rows": report["rows"]["train"],
            "validation_rows": report["rows"]["validation"],
            "test_rows": report["rows"]["test"],
            "date_ranges": report["date_ranges"],
            "validation_amount_mae_mwh_by_candidate": report["validation_amount_mae_mwh"],
            "test": report["test"],
            "test_zero_amount_baseline": report["test_zero_amount_baseline"],
            "test_monthly_median_baseline_mae_mwh": report["test_monthly_median_baseline_mae_mwh"],
            "test_by_quarter": report["test_by_quarter"],
            "future_window_notice": (
                "Accuracy of the live multi-day V2 window is not validated by the fixed-24-hour-lead historical test."
            ),
            "source_caveat": report["source_caveat"],
            "not_comparable_to_v1_mae": report["not_comparable_to_v1_mae"],
        }
        return info

    def dataset_coverage(self) -> dict:
        """Describe the frozen evaluation days separately from requestable forecast dates."""
        if self.bundle is None:
            self.load()
        assert self.bundle is not None
        report = json.loads(self.report_path.read_text(encoding="utf-8"))
        today = datetime.now(timezone.utc).date()
        partitions = {
            split: {
                "first_target_date_utc": date.fromisoformat(report["date_ranges"][split][0][:10]).isoformat(),
                "last_target_date_utc": date.fromisoformat(report["date_ranges"][split][1][:10]).isoformat(),
                "complete_day_count": int(report["rows"][split]),
            }
            for split in ("train", "validation", "test")
        }
        return {
            "model_version": self.bundle["metadata"]["model_version"],
            "historical_date_min_utc": min(part["first_target_date_utc"] for part in partitions.values()),
            "historical_date_max_utc": max(part["last_target_date_utc"] for part in partitions.values()),
            "historical_complete_day_count": sum(part["complete_day_count"] for part in partitions.values()),
            "partitions": partitions,
            "fitted_through_date_utc": self.bundle["metadata"]["trained_through_utc"][:10],
            "requestable_date_min_utc": EARLIEST_TARGET_DATE.isoformat(),
            "requestable_date_max_utc": today.isoformat(),
            "forward_window_date_min_utc": (today + timedelta(days=1)).isoformat(),
            "forward_window_date_max_utc": (today + timedelta(days=7)).isoformat(),
            "maximum_forward_window_days": 7,
            "date_format": "YYYY-MM-DD",
            "timezone": "UTC",
            "dataset_sha256": report["dataset_sha256"],
            "forecast_source": report["source"],
            "request_notice": (
                "The historical dataset is a fixed development/evaluation snapshot, not the live request limit. "
                "A date through today UTC can be requested, but a prediction is not guaranteed if the "
                "Open-Meteo forecast archive lacks all required hourly inputs. "
                "Observed daily outcomes have separate coverage at /actuals/coverage."
            ),
        }

    def predict_features(
        self, features: pd.DataFrame, *, issued_at_utc: datetime | None = None,
    ) -> dict:
        if self.bundle is None:
            self.load()
        assert self.bundle is not None
        if len(features) != 1:
            raise ValueError("Exactly one daily forecast row required")
        target_day = pd.Timestamp(features.iloc[0]["issue_timestamp_utc"])
        issue = pd.Timestamp(issued_at_utc) if issued_at_utc is not None else target_day
        available = pd.Timestamp(features.iloc[0]["forecast_max_available_at_utc"])
        if target_day.tzinfo is None or issue.tzinfo is None or available.tzinfo is None or available > issue:
            raise ValueError("Forecast vintage is not available at issue time")
        columns = self.bundle["metadata"]["feature_columns"]
        if set(columns) - set(features):
            raise ValueError(f"Missing forecast features: {sorted(set(columns) - set(features))}")
        if not np.isfinite(features[columns].to_numpy(dtype=float)).all():
            raise ValueError("Non-finite forecast features")
        probability = float(self.bundle["event_model"].predict_proba(features[columns])[:, 1][0])
        amount = float(_predict_amount(
            self.bundle["metadata"]["selected_amount_method"],
            self.bundle["amount_model"], features[columns], np.array([probability]),
        )[0])
        return {
            "model_version": self.bundle["metadata"]["model_version"],
            "experimental": True,
            "target_date_utc": target_day.date().isoformat(),
            "issue_timestamp_utc": issue.isoformat(),
            "forecast_max_available_at_utc": available.isoformat(),
            "curtailment_event_probability": probability,
            "predicted_curtailment_mwh": amount,
            "target": "Total EirGrid curtailment during this UTC day",
        }

    def predict_date(self, target_date: date) -> dict:
        if target_date < EARLIEST_TARGET_DATE:
            raise ValueError(f"No complete archived forecast before {EARLIEST_TARGET_DATE.isoformat()}")
        if target_date > datetime.now(timezone.utc).date():
            raise ValueError("A future target day has not reached its 00:00 UTC issue time")
        forecast = pd.concat(
            [fetch_previous_runs(region, target_date, target_date, cache=False) for region in REGIONS],
            ignore_index=True,
        )
        features = build_forecast_features(forecast)
        if len(features) != 1:
            raise ValueError("The archived forecast has missing hours or unavailable inputs for this day")
        return self.predict_features(features)

    def predict_forward_window(self, start_date: date, days: int = 7) -> dict:
        """Score complete future UTC days from live, uncached weather forecasts.

        The existing V2 bundle is reused without retraining. Its test metrics
        describe fixed 24-hour-lead archive fields, not these longer leads.
        """

        today = datetime.now(timezone.utc).date()
        if start_date <= today:
            raise ValueError("start_date_utc must be tomorrow UTC or later for complete future days")
        if days < 1 or days > 7:
            raise ValueError("days must be between 1 and 7")
        end_date = start_date + timedelta(days=days - 1)
        if end_date > today + timedelta(days=7):
            raise ValueError("The last target day must be within the next seven UTC days")

        forecasts = pd.concat(
            [fetch_live_forecast(region, start_date, end_date) for region in REGIONS],
            ignore_index=True,
        )
        issued_at = datetime.now(timezone.utc)
        if start_date <= issued_at.date():
            raise ValueError("start_date_utc is no longer a future UTC day; retry with tomorrow UTC")
        try:
            features = build_forecast_features(forecasts)
        except ValueError as error:
            raise OSError("Live forecast inputs are malformed or incomplete") from error
        expected = [start_date + timedelta(days=offset) for offset in range(days)]
        actual = [] if features.empty else list(pd.to_datetime(features["issue_timestamp_utc"], utc=True).dt.date)
        if actual != expected:
            raise OSError("Live forecast lacks complete hourly weather inputs for every requested UTC day")

        predictions = []
        for _, row in features.iterrows():
            prediction = self.predict_features(row.to_frame().T, issued_at_utc=issued_at)
            target_start = datetime.combine(date.fromisoformat(prediction["target_date_utc"]), datetime.min.time(), tzinfo=timezone.utc)
            prediction["forecast_lead_hours"] = round((target_start - issued_at).total_seconds() / 3600, 2)
            predictions.append(prediction)
        return {
            "model_version": predictions[0]["model_version"],
            "semantics": "live_forward_daily_curtailment",
            "experimental": True,
            "notice": (
                "These live GFS forecasts are available at request time, but V2 was trained and "
                "tested on fixed 24-hour-lead archived weather. Accuracy at multi-day leads is "
                "not validated; do not use as a dispatch instruction."
            ),
            "forecast_source": f"Open-Meteo live {FORECAST_MODEL}",
            "issued_at_utc": issued_at.isoformat(),
            "start_date_utc": start_date.isoformat(),
            "end_date_utc": end_date.isoformat(),
            "prediction_count": len(predictions),
            "predictions": predictions,
        }
