"""An opt-in horizon-specific residual model; never overwrites v1.1.0.

Only measurements already present in the v1 issue-time feature contract are
used here. Forecast, frequency and outage research tables lack aligned labels
and cannot honestly be fitted into this candidate yet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

from .constants import SUPPORTED_FORECAST_HORIZONS


PHYSICAL_FEATURES = (
    "v2_wind_unrealised_mw",
    "v2_wind_acceleration_mw",
    "v2_load_acceleration_mw",
    "v2_snsp_headroom_ratio",
)


def candidate_features(frame: pd.DataFrame, *, physical: bool) -> pd.DataFrame:
    """Add issue-row interactions without joining future observations."""

    result = frame.copy()
    if not physical:
        return result
    required = {
        "eirgrid_ie_wind_availability_mw",
        "eirgrid_ie_wind_generation_mw",
        "wind_lag_1",
        "wind_lag_2",
        "eirgrid_ie_demand_mw",
        "load_lag_1",
        "load_lag_2",
        "eirgrid_snsp_ratio",
    }
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"Physical candidate missing issue-time inputs: {missing}")
    result["v2_wind_unrealised_mw"] = (
        frame["eirgrid_ie_wind_availability_mw"]
        - frame["eirgrid_ie_wind_generation_mw"]
    ).clip(lower=0)
    result["v2_wind_acceleration_mw"] = (
        frame["eirgrid_ie_wind_generation_mw"]
        - 2 * frame["wind_lag_1"]
        + frame["wind_lag_2"]
    )
    result["v2_load_acceleration_mw"] = (
        frame["eirgrid_ie_demand_mw"]
        - 2 * frame["load_lag_1"]
        + frame["load_lag_2"]
    )
    result["v2_snsp_headroom_ratio"] = 0.75 - frame["eirgrid_snsp_ratio"]
    return result


class HorizonResidualModel:
    """Separate Extra Trees residual regressors for 30 and 60 minutes."""

    def __init__(
        self,
        columns: list[str],
        *,
        n_estimators: int = 240,
        min_samples_leaf: int = 3,
        max_features: float = 0.75,
        random_state: int = 42,
    ) -> None:
        self.columns = list(columns)
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.models: dict[int, Pipeline] = {}

    def fit(self, frame: pd.DataFrame) -> "HorizonResidualModel":
        for horizon in SUPPORTED_FORECAST_HORIZONS:
            subset = frame.loc[frame["forecast_horizon_minutes"].eq(horizon)]
            if subset.empty:
                raise ValueError(f"No {horizon}-minute training rows")
            pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    (
                        "model",
                        ExtraTreesRegressor(
                            n_estimators=self.n_estimators,
                            min_samples_leaf=self.min_samples_leaf,
                            max_features=self.max_features,
                            n_jobs=-1,
                            random_state=self.random_state,
                        ),
                    ),
                ]
            )
            residual = (
                subset["dispatch_down_mwh"]
                - subset["dispatch_down_mwh_latest_observed"]
            )
            pipeline.fit(subset[self.columns], residual)
            self.models[int(horizon)] = pipeline
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        predictions = np.full(len(frame), np.nan)
        for horizon, model in self.models.items():
            mask = frame["forecast_horizon_minutes"].eq(horizon).to_numpy()
            if mask.any():
                subset = frame.loc[mask]
                predictions[mask] = np.maximum(
                    subset["dispatch_down_mwh_latest_observed"].to_numpy(dtype=float)
                    + model.predict(subset[self.columns]),
                    0.0,
                )
        if not np.isfinite(predictions).all():
            raise ValueError("Candidate prediction has unsupported horizons or non-finite values")
        return predictions


def blend_predictions(
    frame: pd.DataFrame,
    baseline: np.ndarray,
    candidate: np.ndarray,
    weights: dict[str, float],
) -> np.ndarray:
    fractions = frame["forecast_horizon_minutes"].astype(int).astype(str).map(weights)
    if fractions.isna().any() or not fractions.between(0, 1).all():
        raise ValueError("A valid blend weight is required for each horizon")
    values = fractions.to_numpy(dtype=float) * candidate + (1 - fractions.to_numpy(dtype=float)) * baseline
    return np.maximum(values, 0.0)


def select_blend_weights(
    frame: pd.DataFrame,
    baseline: np.ndarray,
    candidate: np.ndarray,
    weight_grid: list[float] | tuple[float, ...] = (0, 0.1, 0.2, 0.3, 0.4, 0.5),
) -> dict[str, float]:
    """Select only on development predictions; favor v1 on exact ties."""

    if not weight_grid or any(not 0 <= weight <= 1 for weight in weight_grid):
        raise ValueError("Blend weights must be within [0, 1]")
    truth = frame["dispatch_down_mwh"].to_numpy(dtype=float)
    result: dict[str, float] = {}
    for horizon in SUPPORTED_FORECAST_HORIZONS:
        mask = frame["forecast_horizon_minutes"].eq(horizon).to_numpy()
        if not mask.any():
            raise ValueError(f"No {horizon}-minute development rows")
        scored: list[tuple[float, float]] = []
        for weight in weight_grid:
            blended = np.maximum((1 - weight) * baseline[mask] + weight * candidate[mask], 0)
            scored.append((float(mean_absolute_error(truth[mask], blended)), float(weight)))
        result[str(horizon)] = min(scored)[1]
    return result
