"""Research-only, point-in-time multi-year dispatch-down experiment.

Historical reports are ex-post. A configurable observation delay is applied to
*every* source field; live publication latency must be verified before serving.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import PROJECT_ROOT


HISTORY_PATH = PROJECT_ROOT / "data" / "processed" / "eirgrid_core_history_30min.csv.gz"
SIGNALS = (
    "dispatch_down_mwh",
    "curtailment_mwh",
    "constraint_mwh",
    "eirgrid_ie_wind_availability_mw",
    "eirgrid_ie_wind_generation_mw",
    "eirgrid_ni_wind_availability_mw",
    "eirgrid_ni_wind_generation_mw",
    "eirgrid_ie_demand_mw",
    "eirgrid_ni_demand_mw",
    "eirgrid_snsp_ratio",
    "eirgrid_all_island_wind_availability_mw",
    "eirgrid_all_island_demand_mw",
    "eirgrid_all_island_solar_availability_mw",
)
TARGET = "dispatch_down_mwh"
HORIZONS = (30, 60)


def source_sha256(path: Path | str = HISTORY_PATH) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_causal_history_features(
    history: pd.DataFrame,
    *,
    observation_delay_hours: int = 24,
) -> pd.DataFrame:
    """Join delayed observations and exact future labels by timestamp, never row offset."""
    if observation_delay_hours < 1:
        raise ValueError("An explicit positive observation delay is required")
    required = {"timestamp_utc", *SIGNALS}
    missing = sorted(required - set(history))
    if missing:
        raise ValueError(f"History is missing required columns: {missing}")
    source = history.loc[:, ["timestamp_utc", *SIGNALS]].copy()
    source["timestamp_utc"] = pd.to_datetime(source["timestamp_utc"], utc=True, errors="raise")
    if source["timestamp_utc"].duplicated().any():
        raise ValueError("History timestamps must be unique")
    source = source.sort_values("timestamp_utc").reset_index(drop=True)
    if not source["timestamp_utc"].is_monotonic_increasing:
        raise ValueError("History must be chronological")

    # Rolling statistics are computed on the source chronology before the as-of
    # join. They contain the delayed snapshot and older values only.
    snapshot = source.rename(columns={"timestamp_utc": "feature_timestamp_utc"}).copy()
    for signal in SIGNALS:
        snapshot[f"{signal}_asof"] = snapshot[signal]
    for signal in (TARGET, "eirgrid_ie_wind_availability_mw", "eirgrid_ie_demand_mw"):
        snapshot[f"{signal}_mean_24h_asof"] = source[signal].rolling(48, min_periods=1).mean()
    feature_names = [column for column in snapshot if column.endswith("_asof")]
    snapshot = snapshot[["feature_timestamp_utc", *feature_names]]

    issues = source[["timestamp_utc"]].rename(columns={"timestamp_utc": "issue_timestamp_utc"})
    issues["latest_eligible_observation_utc"] = (
        issues["issue_timestamp_utc"] - pd.Timedelta(hours=observation_delay_hours)
    )
    features = pd.merge_asof(
        issues,
        snapshot,
        left_on="latest_eligible_observation_utc",
        right_on="feature_timestamp_utc",
        direction="backward",
        allow_exact_matches=True,
    )
    features = features.dropna(subset=["feature_timestamp_utc"])
    features["feature_age_hours"] = (
        features["issue_timestamp_utc"] - features["feature_timestamp_utc"]
    ).dt.total_seconds() / 3600
    clock_hour = features["issue_timestamp_utc"].dt.hour + features["issue_timestamp_utc"].dt.minute / 60
    day_of_year = features["issue_timestamp_utc"].dt.dayofyear
    features["hour_sin"] = np.sin(2 * np.pi * clock_hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * clock_hour / 24)
    features["year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    features["year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    features["weekday"] = features["issue_timestamp_utc"].dt.dayofweek
    for name in feature_names:
        features[f"{name}_available"] = features[name].notna().astype("int8")

    label_lookup = source.set_index("timestamp_utc")[TARGET]
    outputs = []
    for horizon in HORIZONS:
        frame = features.copy()
        frame["forecast_horizon_minutes"] = horizon
        frame["target_timestamp_utc"] = frame["issue_timestamp_utc"] + pd.Timedelta(minutes=horizon)
        frame["dispatch_down_mwh"] = label_lookup.reindex(frame["target_timestamp_utc"]).to_numpy()
        outputs.append(frame.dropna(subset=["dispatch_down_mwh"]))
    result = pd.concat(outputs, ignore_index=True).sort_values(
        ["issue_timestamp_utc", "forecast_horizon_minutes"]
    ).reset_index(drop=True)
    if result.duplicated(["issue_timestamp_utc", "forecast_horizon_minutes"]).any():
        raise ValueError("Duplicate issue/horizon keys")
    if not (result["feature_timestamp_utc"] <= result["latest_eligible_observation_utc"]).all():
        raise ValueError("A feature crosses its observation cutoff")
    return result


def model_feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "issue_timestamp_utc",
        "target_timestamp_utc",
        "feature_timestamp_utc",
        "latest_eligible_observation_utc",
        "dispatch_down_mwh",
    }
    columns = [column for column in frame if column not in excluded]
    if not all(pd.api.types.is_numeric_dtype(frame[column]) for column in columns):
        raise ValueError("Model features must be numeric")
    return columns
