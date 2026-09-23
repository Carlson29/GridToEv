from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from .constants import (
    DEFAULT_DATASET_PATH,
    DEFAULT_MODEL_PATH,
    INTERVAL_HOURS,
    SUPPORTED_FORECAST_HORIZONS,
)


class FeatureValidationError(ValueError):
    """Raised when a live feature snapshot does not match the training contract."""


class DatasetSelectionError(LookupError):
    """Raised with structured guidance when a requested dataset interval is unavailable."""

    def __init__(self, detail: dict[str, Any]) -> None:
        super().__init__(detail["message"])
        self.detail = detail


class PredictionService:
    def __init__(
        self,
        model_path: Path | str = DEFAULT_MODEL_PATH,
        dataset_path: Path | str | None = DEFAULT_DATASET_PATH,
    ) -> None:
        self.model_path = Path(model_path).resolve()
        self.dataset_path = Path(dataset_path).resolve() if dataset_path else None
        self.bundle: dict[str, Any] | None = None
        self.dataset: pd.DataFrame | None = None

    @property
    def loaded(self) -> bool:
        return self.bundle is not None

    @property
    def metadata(self) -> dict[str, Any]:
        if self.bundle is None:
            raise RuntimeError("Prediction service is not loaded")
        return self.bundle["metadata"]

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model bundle not found at {self.model_path}. Run `gridtoev-train` first."
            )
        bundle = joblib.load(self.model_path)
        required_keys = {"metadata", "metrics", "feature_columns", "models"}
        if not required_keys.issubset(bundle):
            raise ValueError("Model bundle is missing required keys")
        self.bundle = bundle

        if self.dataset_path is not None:
            if not self.dataset_path.exists():
                raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")
            dataset = pd.read_csv(self.dataset_path)
            dataset["issue_timestamp_utc"] = pd.to_datetime(
                dataset["issue_timestamp_utc"], errors="raise", utc=True
            )
            dataset["target_timestamp_utc"] = pd.to_datetime(
                dataset["target_timestamp_utc"], errors="raise", utc=True
            )
            self.dataset = dataset.sort_values(
                ["issue_timestamp_utc", "forecast_horizon_minutes"]
            ).reset_index(drop=True)

    def model_info(self) -> dict[str, Any]:
        info = dict(self.metadata)
        info.pop("feature_columns", None)
        # Do not expose an absolute server filesystem path through a public API.
        info["model_artifact"] = self.model_path.name
        info["dataset_loaded"] = self.dataset is not None
        info["available_issue_timestamp_min_utc"] = None
        info["available_issue_timestamp_max_utc"] = None
        if self.dataset is not None:
            info["available_issue_timestamp_min_utc"] = self.dataset[
                "issue_timestamp_utc"
            ].min().isoformat()
            info["available_issue_timestamp_max_utc"] = self.dataset[
                "issue_timestamp_utc"
            ].max().isoformat()
        return info

    def available_times(self, limit: int = 96) -> list[str]:
        if self.dataset is None:
            return []
        times = self.dataset["issue_timestamp_utc"].drop_duplicates().sort_values()
        return [timestamp.isoformat() for timestamp in times.tail(limit)]

    def dataset_info(self) -> dict[str, Any]:
        if self.dataset is None:
            raise RuntimeError("No modelling dataset is loaded")
        times = self.dataset["issue_timestamp_utc"].drop_duplicates().sort_values()
        first = times.iloc[0]
        last = times.iloc[-1]
        return {
            "available_issue_timestamp_min_utc": first.isoformat(),
            "available_issue_timestamp_max_utc": last.isoformat(),
            "available_issue_timestamp_count": int(len(times)),
            "available_date_min_utc": first.date().isoformat(),
            "available_date_max_utc": last.date().isoformat(),
            "timezone": "UTC",
            "timestamp_format": "ISO 8601 with timezone: YYYY-MM-DDTHH:MM:SSZ",
            "timestamp_example": first.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "interval_minutes": 30,
            "required_minute_values": [0, 30],
            "supported_forecast_horizons_minutes": list(SUPPORTED_FORECAST_HORIZONS),
            "maximum_window_hours": 48.0,
            "window_semantics": "historical_rolling_short_horizon",
            "window_notice": (
                "A window replays 30/60-minute forecasts at each historical half-hour. "
                "It is not a single 2-hour or day-ahead model forecast."
            ),
        }

    def _selection_detail(
        self,
        timestamp: pd.Timestamp,
        forecast_horizon_minutes: int,
    ) -> dict[str, Any]:
        if self.dataset is None:
            raise RuntimeError("No modelling dataset is loaded")
        horizon_times = self.dataset.loc[
            self.dataset["forecast_horizon_minutes"].eq(forecast_horizon_minutes),
            "issue_timestamp_utc",
        ].drop_duplicates().sort_values()
        before = horizon_times.loc[horizon_times.lt(timestamp)]
        after = horizon_times.loc[horizon_times.gt(timestamp)]
        info = self.dataset_info()
        return {
            "error": "dataset_timestamp_not_available",
            "message": (
                f"No dataset row exists for {timestamp.isoformat()} at "
                f"the {forecast_horizon_minutes}-minute horizon."
            ),
            "requested_issue_timestamp_utc": timestamp.isoformat(),
            "requested_forecast_horizon_minutes": int(forecast_horizon_minutes),
            "expected_format": info["timestamp_format"],
            "timestamp_example": info["timestamp_example"],
            "available_issue_timestamp_min_utc": info[
                "available_issue_timestamp_min_utc"
            ],
            "available_issue_timestamp_max_utc": info[
                "available_issue_timestamp_max_utc"
            ],
            "nearest_before_utc": (
                before.iloc[-1].isoformat() if not before.empty else None
            ),
            "nearest_after_utc": (
                after.iloc[0].isoformat() if not after.empty else None
            ),
            "available_times_endpoint": "/dataset/available-times",
        }

    def _ensure_loaded(self) -> dict[str, Any]:
        if self.bundle is None:
            raise RuntimeError("Prediction service is not loaded")
        return self.bundle

    def _predict_frame(
        self,
        frame: pd.DataFrame,
        flexible_load_capacity_mw: float,
        issue_timestamp_utc: pd.Timestamp,
        forecast_horizon_minutes: int,
    ) -> dict[str, Any]:
        bundle = self._ensure_loaded()
        models = bundle["models"]
        columns = bundle["feature_columns"]
        X = frame.reindex(columns=columns)

        probability = float(models["event_classifier"].predict_proba(X)[0, 1])
        latest_observed = max(
            float(X["dispatch_down_mwh_latest_observed"].iloc[0]),
            0.0,
        )
        previous_observed = max(float(X["dispatch_down_mwh_lag_1"].iloc[0]), 0.0)
        horizon_key = str(int(forecast_horizon_minutes))
        trend_alpha = float(
            self.metadata["dispatch_trend_alpha_by_horizon"][horizon_key]
        )
        trend_baseline = max(
            latest_observed + trend_alpha * (latest_observed - previous_observed),
            0.0,
        )
        predicted_change = float(models["dispatch_down_regressor"].predict(X)[0])
        ml_total = max(latest_observed + predicted_change, 0.0)
        ml_weight = float(
            self.metadata["dispatch_regression_ml_weight_by_horizon"][horizon_key]
        )
        total = max(
            ml_weight * ml_total + (1.0 - ml_weight) * trend_baseline,
            0.0,
        )
        raw_curtailment = max(float(models["curtailment_regressor"].predict(X)[0]), 0.0)
        raw_constraint = max(float(models["constraint_regressor"].predict(X)[0]), 0.0)

        raw_component_sum = raw_curtailment + raw_constraint
        if total <= 0:
            curtailment = 0.0
            constraint = 0.0
        elif raw_component_sum > 0:
            curtailment = total * raw_curtailment / raw_component_sum
            constraint = total - curtailment
        else:
            shares = self.metadata["default_component_shares"]
            curtailment = total * float(shares["curtailment"])
            constraint = total - curtailment

        raw_quantiles = [
            max(
                latest_observed
                + float(models[f"dispatch_down_quantile_{quantile}"].predict(X)[0]),
                0.0,
            )
            for quantile in ("p10", "p50", "p90")
        ]
        p10, p50, p90 = sorted(raw_quantiles)
        interval_adjustment = float(self.metadata["prediction_interval_adjustment_mwh"])
        p10 = max(p10 - interval_adjustment, 0.0)
        p90 = p90 + interval_adjustment

        threshold = float(self.metadata["classification_threshold"])
        if probability >= 0.70:
            risk_level = "high"
        elif probability >= 0.40:
            risk_level = "medium"
        else:
            risk_level = "low"

        flexible_energy_mwh = flexible_load_capacity_mw * INTERVAL_HOURS
        target_timestamp = issue_timestamp_utc + pd.Timedelta(minutes=forecast_horizon_minutes)
        return {
            "model_version": self.metadata["model_version"],
            "issue_timestamp_utc": issue_timestamp_utc.isoformat(),
            "target_timestamp_utc": target_timestamp.isoformat(),
            "forecast_horizon_minutes": int(forecast_horizon_minutes),
            "dispatch_down_probability": probability,
            "dispatch_down_event_prediction": bool(probability >= threshold),
            "classification_threshold": threshold,
            "risk_level": risk_level,
            "predicted_dispatch_down_mwh": total,
            "predicted_curtailment_mwh": curtailment,
            "predicted_constraint_mwh": constraint,
            "prediction_interval_p10_mwh": p10,
            "prediction_interval_p50_mwh": p50,
            "prediction_interval_p90_mwh": p90,
            "flexible_load_capacity_mw": float(flexible_load_capacity_mw),
            "recoverable_surplus_mwh": min(total, flexible_energy_mwh),
        }

    def predict_from_dataset(
        self,
        issue_timestamp_utc: str | pd.Timestamp,
        forecast_horizon_minutes: int,
        flexible_load_capacity_mw: float = 100.0,
    ) -> dict[str, Any]:
        if forecast_horizon_minutes not in SUPPORTED_FORECAST_HORIZONS:
            raise FeatureValidationError(
                f"forecast_horizon_minutes must be one of {SUPPORTED_FORECAST_HORIZONS}"
            )
        if flexible_load_capacity_mw <= 0:
            raise FeatureValidationError("flexible_load_capacity_mw must be positive")
        if self.dataset is None:
            raise RuntimeError("No modelling dataset is loaded")

        timestamp = pd.Timestamp(issue_timestamp_utc)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        matches = self.dataset.loc[
            self.dataset["issue_timestamp_utc"].eq(timestamp)
            & self.dataset["forecast_horizon_minutes"].eq(forecast_horizon_minutes)
        ]
        if matches.empty:
            raise DatasetSelectionError(
                self._selection_detail(timestamp, forecast_horizon_minutes)
            )
        return self._predict_frame(
            matches.iloc[[0]],
            flexible_load_capacity_mw,
            timestamp,
            forecast_horizon_minutes,
        )

    def predict_latest(self, flexible_load_capacity_mw: float = 100.0) -> list[dict[str, Any]]:
        if self.dataset is None:
            raise RuntimeError("No modelling dataset is loaded")
        common_times: set[pd.Timestamp] | None = None
        for horizon in SUPPORTED_FORECAST_HORIZONS:
            horizon_times = set(
                self.dataset.loc[
                    self.dataset["forecast_horizon_minutes"].eq(horizon),
                    "issue_timestamp_utc",
                ]
            )
            common_times = horizon_times if common_times is None else common_times & horizon_times
        if not common_times:
            raise KeyError("No issue timestamp is available for every supported horizon")
        latest = max(common_times)
        return [
            self.predict_from_dataset(latest, horizon, flexible_load_capacity_mw)
            for horizon in SUPPORTED_FORECAST_HORIZONS
        ]

    def predict_window_from_dataset(
        self,
        start_timestamp_utc: str | pd.Timestamp,
        duration_hours: float,
        forecast_horizons_minutes: list[int],
        flexible_load_capacity_mw: float = 100.0,
    ) -> dict[str, Any]:
        """Replay short-horizon predictions over a historical half-hour window."""
        if self.dataset is None:
            raise RuntimeError("No modelling dataset is loaded")
        if duration_hours < 0.5 or duration_hours > 48:
            raise FeatureValidationError("duration_hours must be between 0.5 and 48")
        steps_float = duration_hours * 2
        if not np.isclose(steps_float, round(steps_float)):
            raise FeatureValidationError("duration_hours must be in 0.5-hour increments")
        horizons = sorted(set(int(value) for value in forecast_horizons_minutes))
        invalid_horizon = any(
            value not in SUPPORTED_FORECAST_HORIZONS for value in horizons
        )
        if not horizons or invalid_horizon:
            raise FeatureValidationError(
                f"forecast_horizons_minutes must contain values from {SUPPORTED_FORECAST_HORIZONS}"
            )
        if flexible_load_capacity_mw <= 0:
            raise FeatureValidationError("flexible_load_capacity_mw must be positive")

        start = pd.Timestamp(start_timestamp_utc)
        if start.tzinfo is None:
            start = start.tz_localize("UTC")
        else:
            start = start.tz_convert("UTC")
        steps = int(round(steps_float))
        expected_times = pd.date_range(start=start, periods=steps, freq="30min")

        common_times: set[pd.Timestamp] | None = None
        for horizon in horizons:
            horizon_times = set(
                self.dataset.loc[
                    self.dataset["forecast_horizon_minutes"].eq(horizon),
                    "issue_timestamp_utc",
                ]
            )
            common_times = (
                horizon_times if common_times is None else common_times & horizon_times
            )
        common_times = common_times or set()
        missing_times = [timestamp for timestamp in expected_times if timestamp not in common_times]
        if missing_times:
            info = self.dataset_info()
            latest_valid_start = None
            if common_times:
                candidate = max(common_times) - pd.Timedelta(minutes=30 * (steps - 1))
                if candidate >= min(common_times):
                    latest_valid_start = candidate.isoformat()
            raise DatasetSelectionError(
                {
                    "error": "dataset_window_not_available",
                    "message": (
                        "The requested window is not fully contained in the dataset for every "
                        "requested horizon."
                    ),
                    "requested_start_timestamp_utc": start.isoformat(),
                    "requested_duration_hours": float(duration_hours),
                    "requested_forecast_horizons_minutes": horizons,
                    "first_missing_issue_timestamp_utc": missing_times[0].isoformat(),
                    "latest_valid_start_utc_for_duration": latest_valid_start,
                    "expected_format": info["timestamp_format"],
                    "available_issue_timestamp_min_utc": info[
                        "available_issue_timestamp_min_utc"
                    ],
                    "available_issue_timestamp_max_utc": info[
                        "available_issue_timestamp_max_utc"
                    ],
                }
            )

        predictions = [
            self.predict_from_dataset(timestamp, horizon, flexible_load_capacity_mw)
            for timestamp in expected_times
            for horizon in horizons
        ]
        summary_by_horizon: dict[str, dict[str, Any]] = {}
        for horizon in horizons:
            items = [
                item
                for item in predictions
                if item["forecast_horizon_minutes"] == horizon
            ]
            summary_by_horizon[str(horizon)] = {
                "interval_count": len(items),
                "average_dispatch_down_probability": float(
                    np.mean([item["dispatch_down_probability"] for item in items])
                ),
                "total_predicted_dispatch_down_mwh": float(
                    sum(item["predicted_dispatch_down_mwh"] for item in items)
                ),
                "peak_predicted_dispatch_down_mwh": float(
                    max(item["predicted_dispatch_down_mwh"] for item in items)
                ),
                "total_recoverable_surplus_mwh": float(
                    sum(item["recoverable_surplus_mwh"] for item in items)
                ),
            }

        return {
            "model_version": self.metadata["model_version"],
            "semantics": "historical_rolling_short_horizon",
            "notice": self.dataset_info()["window_notice"],
            "start_timestamp_utc": start.isoformat(),
            "end_timestamp_exclusive_utc": (
                start + pd.Timedelta(hours=duration_hours)
            ).isoformat(),
            "duration_hours": float(duration_hours),
            "interval_minutes": 30,
            "forecast_horizons_minutes": horizons,
            "prediction_count": len(predictions),
            "summary_by_horizon": summary_by_horizon,
            "predictions": predictions,
        }

    def predict_features(
        self,
        features: Mapping[str, float],
        forecast_horizon_minutes: int,
        flexible_load_capacity_mw: float = 100.0,
        issue_timestamp_utc: str | pd.Timestamp | None = None,
    ) -> dict[str, Any]:
        if forecast_horizon_minutes not in SUPPORTED_FORECAST_HORIZONS:
            raise FeatureValidationError(
                f"forecast_horizon_minutes must be one of {SUPPORTED_FORECAST_HORIZONS}"
            )
        bundle = self._ensure_loaded()
        required = set(bundle["feature_columns"]) - {"forecast_horizon_minutes"}
        missing = sorted(required - set(features))
        if missing:
            raise FeatureValidationError(
                f"Missing {len(missing)} required features. First missing fields: {missing[:10]}"
            )

        row = {column: float(features[column]) for column in required}
        row["forecast_horizon_minutes"] = forecast_horizon_minutes
        frame = pd.DataFrame([row])
        timestamp = pd.Timestamp(issue_timestamp_utc or pd.Timestamp.now(tz="UTC"))
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        return self._predict_frame(
            frame,
            flexible_load_capacity_mw,
            timestamp,
            forecast_horizon_minutes,
        )
