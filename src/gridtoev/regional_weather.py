"""Capacity-weighted, publication-safe regional weather features.

Weather forecasts are model-run vintages.  A run is eligible only after a
conservative six-hour processing delay, and only values from eligible runs may
be selected for a model issue time.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

from gridtoev.constants import IDENTIFIER_COLUMNS, TARGET_COLUMNS
from gridtoev.eirgrid_history import DataQualityError


SCHEMA_VERSION = "regional-weather-v1"
MODEL_NAME = "ecmwf_ifs"
FORECAST_PUBLICATION_LAG_HOURS = 6
ANALYSIS_AVAILABILITY_LAG_HOURS = 6

API_TO_COLUMN = {
    "wind_speed_100m": "wind_speed_100m_mps",
    "wind_direction_100m": "wind_direction_100m_deg",
    "wind_gusts_10m": "wind_gusts_10m_mps",
    "temperature_2m": "temperature_2m_c",
    "pressure_msl": "pressure_msl_hpa",
    "cloud_cover": "cloud_cover_pct",
    "shortwave_radiation": "shortwave_radiation_wm2",
}
WEATHER_VALUE_COLUMNS = list(API_TO_COLUMN.values())

FORECAST_COLUMNS = [
    "region",
    "latitude",
    "longitude",
    "model",
    "run_initialized_at_utc",
    "published_at_utc",
    "valid_timestamp_utc",
    "lead_hours",
    *WEATHER_VALUE_COLUMNS,
    "downloaded_at_utc",
    "source_url",
    "schema_version",
]

ANALYSIS_COLUMNS = [
    "region",
    "latitude",
    "longitude",
    "valid_timestamp_utc",
    "published_at_utc",
    *WEATHER_VALUE_COLUMNS,
    "source_role",
    "downloaded_at_utc",
    "source_url",
    "schema_version",
]


def _utc(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _payload_list(payload: object) -> list[dict[str, object]]:
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
        return payload
    raise DataQualityError("Open-Meteo payload must be an object or list of objects")


def validate_region_weights(regions: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the committed regional capacity proxy."""

    required = {
        "region",
        "latitude",
        "longitude",
        "wind_capacity_mw",
        "solar_capacity_mw",
    }
    missing = sorted(required - set(regions.columns))
    if missing:
        raise DataQualityError(f"Regional weather weights are missing columns: {missing}")
    result = regions.copy()
    result["region"] = result["region"].astype(str).str.strip().str.lower()
    if result["region"].duplicated().any():
        raise DataQualityError("Regional weather weights contain duplicate regions")
    if not result["region"].str.fullmatch(r"[a-z0-9_]+", na=False).all():
        raise DataQualityError("Region names must use lowercase letters, digits, or underscores")
    for column in (
        "latitude",
        "longitude",
        "wind_capacity_mw",
        "solar_capacity_mw",
    ):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    numeric = result[
        ["latitude", "longitude", "wind_capacity_mw", "solar_capacity_mw"]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise DataQualityError("Regional weather weights contain non-finite values")
    if not result["latitude"].between(-90, 90).all():
        raise DataQualityError("Regional latitudes must be between -90 and 90")
    if not result["longitude"].between(-180, 180).all():
        raise DataQualityError("Regional longitudes must be between -180 and 180")
    if (result[["wind_capacity_mw", "solar_capacity_mw"]] < 0).any().any():
        raise DataQualityError("Regional capacity values cannot be negative")
    wind_total = float(result["wind_capacity_mw"].sum())
    solar_total = float(result["solar_capacity_mw"].sum())
    if wind_total <= 0 or solar_total <= 0:
        raise DataQualityError("Wind and solar capacity totals must both be positive")
    result["wind_capacity_weight"] = result["wind_capacity_mw"] / wind_total
    result["solar_capacity_weight"] = result["solar_capacity_mw"] / solar_total
    return result


def _parse_hourly_payloads(
    payload: object,
    regions: pd.DataFrame,
) -> list[tuple[pd.Series, pd.DataFrame]]:
    region_table = validate_region_weights(regions).reset_index(drop=True)
    payloads = _payload_list(payload)
    if len(payloads) != len(region_table):
        raise DataQualityError(
            "Open-Meteo response count does not match requested regional points"
        )
    parsed: list[tuple[pd.Series, pd.DataFrame]] = []
    for index, document in enumerate(payloads):
        hourly = document.get("hourly")
        if not isinstance(hourly, dict) or "time" not in hourly:
            raise DataQualityError("Open-Meteo response is missing hourly.time")
        frame = pd.DataFrame(hourly)
        absent = sorted(set(API_TO_COLUMN) - set(frame.columns))
        if absent:
            raise DataQualityError(f"Open-Meteo response is missing variables: {absent}")
        frame = frame.rename(columns=API_TO_COLUMN)
        frame["valid_timestamp_utc"] = pd.to_datetime(
            frame.pop("time"), utc=True, errors="raise"
        )
        for column in WEATHER_VALUE_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        parsed.append((region_table.iloc[index], frame))
    return parsed


def parse_open_meteo_single_run(
    payload: object,
    regions: pd.DataFrame,
    *,
    run_initialized_at: object,
    downloaded_at: object,
    source_url: str,
    publication_lag_hours: int = FORECAST_PUBLICATION_LAG_HOURS,
) -> pd.DataFrame:
    """Normalize a multi-location Open-Meteo Single Runs response."""

    run = _utc(run_initialized_at)
    published = run + pd.Timedelta(hours=publication_lag_hours)
    downloaded = _utc(downloaded_at)
    frames: list[pd.DataFrame] = []
    for region, frame in _parse_hourly_payloads(payload, regions):
        result = frame.copy()
        result["region"] = region["region"]
        result["latitude"] = float(region["latitude"])
        result["longitude"] = float(region["longitude"])
        result["model"] = MODEL_NAME
        result["run_initialized_at_utc"] = run
        result["published_at_utc"] = published
        result["lead_hours"] = (
            result["valid_timestamp_utc"] - run
        ).dt.total_seconds() / 3600
        result["downloaded_at_utc"] = downloaded
        result["source_url"] = source_url
        result["schema_version"] = SCHEMA_VERSION
        frames.append(result)
    return pd.concat(frames, ignore_index=True)[FORECAST_COLUMNS]


def parse_open_meteo_analysis(
    payload: object,
    regions: pd.DataFrame,
    *,
    downloaded_at: object,
    source_url: str,
    availability_lag_hours: int = ANALYSIS_AVAILABILITY_LAG_HOURS,
) -> pd.DataFrame:
    """Normalize stitched historical forecasts as a delayed analysis proxy."""

    downloaded = _utc(downloaded_at)
    frames: list[pd.DataFrame] = []
    for region, frame in _parse_hourly_payloads(payload, regions):
        result = frame.copy()
        result["region"] = region["region"]
        result["latitude"] = float(region["latitude"])
        result["longitude"] = float(region["longitude"])
        result["published_at_utc"] = result["valid_timestamp_utc"] + pd.Timedelta(
            hours=availability_lag_hours
        )
        result["source_role"] = "analysis_proxy"
        result["downloaded_at_utc"] = downloaded
        result["source_url"] = source_url
        result["schema_version"] = SCHEMA_VERSION
        frames.append(result)
    return pd.concat(frames, ignore_index=True)[ANALYSIS_COLUMNS]


def _prepare_forecast_grid(forecasts: pd.DataFrame) -> pd.DataFrame:
    if forecasts.empty:
        return pd.DataFrame(columns=FORECAST_COLUMNS)
    result = forecasts.copy()
    for column in (
        "run_initialized_at_utc",
        "published_at_utc",
        "valid_timestamp_utc",
        "downloaded_at_utc",
    ):
        result[column] = pd.to_datetime(result[column], utc=True, errors="raise")
    if result.duplicated(
        ["region", "run_initialized_at_utc", "valid_timestamp_utc"]
    ).any():
        raise DataQualityError("Weather forecast vintages contain duplicate rows")

    grids: list[pd.DataFrame] = []
    group_columns = ["region", "run_initialized_at_utc", "published_at_utc"]
    for keys, group in result.groupby(group_columns, sort=True):
        group = group.sort_values("valid_timestamp_utc").set_index(
            "valid_timestamp_utc"
        )
        index = pd.date_range(group.index.min(), group.index.max(), freq="30min")
        numeric = group[WEATHER_VALUE_COLUMNS].reindex(index).interpolate(
            method="time", limit_area="inside"
        )
        numeric["region"] = keys[0]
        numeric["run_initialized_at_utc"] = keys[1]
        numeric["published_at_utc"] = keys[2]
        numeric["valid_timestamp_utc"] = index
        numeric["pressure_tendency_3h_hpa"] = (
            numeric["pressure_msl_hpa"] - numeric["pressure_msl_hpa"].shift(6)
        )
        direction_radians = np.deg2rad(numeric["wind_direction_100m_deg"])
        numeric["wind_u_100m_mps"] = -numeric["wind_speed_100m_mps"] * np.sin(
            direction_radians
        )
        numeric["wind_v_100m_mps"] = -numeric["wind_speed_100m_mps"] * np.cos(
            direction_radians
        )
        numeric["wind_gust_excess_mps"] = (
            numeric["wind_gusts_10m_mps"] - numeric["wind_speed_100m_mps"]
        )
        grids.append(numeric.reset_index(drop=True))
    return pd.concat(grids, ignore_index=True)


def _weighted_mean(
    frame: pd.DataFrame,
    value_column: str,
    weight_column: str,
) -> float:
    values = pd.to_numeric(frame[value_column], errors="coerce")
    weights = pd.to_numeric(frame[weight_column], errors="coerce")
    usable = values.notna() & weights.gt(0)
    if not usable.any():
        return np.nan
    return float(np.average(values.loc[usable], weights=weights.loc[usable]))


def _weighted_std(
    frame: pd.DataFrame,
    value_column: str,
    weight_column: str,
) -> float:
    values = pd.to_numeric(frame[value_column], errors="coerce")
    weights = pd.to_numeric(frame[weight_column], errors="coerce")
    usable = values.notna() & weights.gt(0)
    if not usable.any():
        return np.nan
    values = values.loc[usable].to_numpy(dtype=float)
    weights = weights.loc[usable].to_numpy(dtype=float)
    mean = np.average(values, weights=weights)
    return float(np.sqrt(np.average((values - mean) ** 2, weights=weights)))


def _weather_error_history(
    forecasts: pd.DataFrame,
    analysis: pd.DataFrame,
    regions: pd.DataFrame,
) -> pd.DataFrame:
    if forecasts.empty or analysis.empty:
        return pd.DataFrame()
    forecast = forecasts.copy()
    observed = analysis.copy()
    for frame in (forecast, observed):
        frame["valid_timestamp_utc"] = pd.to_datetime(
            frame["valid_timestamp_utc"], utc=True, errors="raise"
        )
        frame["published_at_utc"] = pd.to_datetime(
            frame["published_at_utc"], utc=True, errors="raise"
        )
    eligible = forecast.loc[
        forecast["published_at_utc"].le(forecast["valid_timestamp_utc"])
    ].copy()
    eligible = eligible.sort_values("published_at_utc").drop_duplicates(
        ["region", "valid_timestamp_utc"], keep="last"
    )
    joined = observed.merge(
        eligible,
        on=["region", "valid_timestamp_utc"],
        how="inner",
        suffixes=("_observed", "_forecast"),
        validate="one_to_one",
    )
    if joined.empty:
        return pd.DataFrame()
    weights = validate_region_weights(regions)[
        ["region", "wind_capacity_weight", "solar_capacity_weight"]
    ]
    joined = joined.merge(weights, on="region", how="left", validate="many_to_one")
    for column in (
        "wind_speed_100m_mps",
        "temperature_2m_c",
        "shortwave_radiation_wm2",
    ):
        joined[f"{column}_error"] = (
            joined[f"{column}_observed"] - joined[f"{column}_forecast"]
        )

    rows: list[dict[str, object]] = []
    for valid, group in joined.groupby("valid_timestamp_utc", sort=True):
        rows.append(
            {
                "valid_timestamp_utc": valid,
                "observed_at_utc": group["published_at_utc_observed"].max(),
                "wind_error_mps": _weighted_mean(
                    group, "wind_speed_100m_mps_error", "wind_capacity_weight"
                ),
                "temperature_error_c": _weighted_mean(
                    group, "temperature_2m_c_error", "solar_capacity_weight"
                ),
                "irradiance_error_wm2": _weighted_mean(
                    group, "shortwave_radiation_wm2_error", "solar_capacity_weight"
                ),
            }
        )
    return pd.DataFrame(rows)


def _error_features(
    history: pd.DataFrame,
    issue: pd.Timestamp,
) -> dict[str, object]:
    empty: dict[str, object] = {
        "weather_error_observed_at_utc": pd.NaT,
        "weather_error_source_available_flag": 0,
        "weather_wind_error_latest_mps": np.nan,
        "weather_wind_error_bias_24h_mps": np.nan,
        "weather_wind_error_mae_24h_mps": np.nan,
        "weather_temperature_error_bias_24h_c": np.nan,
        "weather_irradiance_error_bias_24h_wm2": np.nan,
        "weather_irradiance_error_mae_24h_wm2": np.nan,
        "weather_error_observation_count_24h": 0,
    }
    if history.empty:
        return empty
    known = history.loc[history["observed_at_utc"].le(issue)].copy()
    if known.empty:
        return empty
    latest = known.sort_values("valid_timestamp_utc").iloc[-1]
    window = known.loc[known["valid_timestamp_utc"].ge(issue - pd.Timedelta(hours=24))]
    if window.empty:
        window = known.tail(1)
    empty.update(
        {
            "weather_error_observed_at_utc": latest["observed_at_utc"],
            "weather_error_source_available_flag": 1,
            "weather_wind_error_latest_mps": latest["wind_error_mps"],
            "weather_wind_error_bias_24h_mps": float(window["wind_error_mps"].mean()),
            "weather_wind_error_mae_24h_mps": float(window["wind_error_mps"].abs().mean()),
            "weather_temperature_error_bias_24h_c": float(
                window["temperature_error_c"].mean()
            ),
            "weather_irradiance_error_bias_24h_wm2": float(
                window["irradiance_error_wm2"].mean()
            ),
            "weather_irradiance_error_mae_24h_wm2": float(
                window["irradiance_error_wm2"].abs().mean()
            ),
            "weather_error_observation_count_24h": int(len(window)),
        }
    )
    return empty


def validate_weather_feature_matrix(table: pd.DataFrame) -> dict[str, int]:
    required = {
        "issue_timestamp_utc",
        "target_timestamp_utc",
        "forecast_horizon_minutes",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise DataQualityError(f"Weather feature table is missing columns: {missing}")
    frame = table.copy()
    issue = pd.to_datetime(frame["issue_timestamp_utc"], utc=True, errors="raise")
    target = pd.to_datetime(frame["target_timestamp_utc"], utc=True, errors="raise")
    expected = issue + pd.to_timedelta(frame["forecast_horizon_minutes"], unit="m")
    checks = {
        "duplicate_natural_keys": int(
            frame.duplicated(["issue_timestamp_utc", "forecast_horizon_minutes"]).sum()
        ),
        "horizon_alignment_violations": int(target.ne(expected).sum()),
        "future_publication_violations": 0,
        "nonfinite_numeric_cells": 0,
    }
    for column in frame.columns:
        if column.endswith(("_published_at_utc", "_observed_at_utc")):
            timestamp = pd.to_datetime(frame[column], utc=True, errors="coerce")
            checks["future_publication_violations"] += int(
                (timestamp.notna() & timestamp.gt(issue)).sum()
            )
    numeric = frame.select_dtypes(include=["number"])
    if not numeric.empty:
        checks["nonfinite_numeric_cells"] = int(
            np.isinf(numeric.to_numpy(dtype=float)).sum()
        )
    failed = {name: value for name, value in checks.items() if value}
    if failed:
        raise DataQualityError(f"Weather feature validation failed: {failed}")
    return checks


def _dictionary(table: pd.DataFrame) -> pd.DataFrame:
    identifiers = {
        "issue_timestamp_utc",
        "target_timestamp_utc",
        "forecast_horizon_minutes",
    }
    rows: list[dict[str, object]] = []
    for column in table.columns:
        role = "identifier" if column in identifiers else "feature"
        if column.endswith("_mps"):
            unit = "m/s"
        elif column.endswith("_hpa"):
            unit = "hPa"
        elif column.endswith("_c"):
            unit = "degrees C"
        elif column.endswith("_wm2"):
            unit = "W/m2"
        elif column.endswith("_pct"):
            unit = "percent"
        elif column.endswith("_hours"):
            unit = "hours"
        elif column.endswith("_utc"):
            unit = "UTC timestamp"
        elif column.endswith(("_flag", "_count_24h")):
            unit = "count"
        else:
            unit = "ratio or unitless"
        if "error" in column:
            source = "Open-Meteo ECMWF run minus delayed historical-forecast analysis proxy"
        elif column in identifiers:
            source = "model row"
        else:
            source = "Open-Meteo ECMWF IFS single-run archive"
        rows.append(
            {
                "column": column,
                "role": role,
                "dtype": str(table[column].dtype),
                "unit": unit,
                "source": source,
                "formula": "See docs/REGIONAL_WEATHER_SIGNALS.md.",
                "availability_rule": (
                    "Newest regional run conservatively available by issue time; "
                    "errors require the delayed analysis proxy to be available."
                ),
                "missingness_handling": (
                    "Timestamps remain null; matching availability flags are zero."
                    if column.endswith("_utc")
                    else "Null for unavailable values; explicit regional/source flags are provided."
                ),
            }
        )
    return pd.DataFrame(rows)


def build_regional_weather_features(
    modelling_rows: pd.DataFrame,
    forecasts: pd.DataFrame,
    analysis: pd.DataFrame,
    regions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Create one regional-weather feature row per issue time and horizon."""

    weights = validate_region_weights(regions)
    result = modelling_rows[
        ["issue_timestamp_utc", "target_timestamp_utc", "forecast_horizon_minutes"]
    ].copy()
    result["issue_timestamp_utc"] = pd.to_datetime(
        result["issue_timestamp_utc"], utc=True, errors="raise"
    )
    result["target_timestamp_utc"] = pd.to_datetime(
        result["target_timestamp_utc"], utc=True, errors="raise"
    )
    if result.duplicated(["issue_timestamp_utc", "forecast_horizon_minutes"]).any():
        raise DataQualityError("Modelling rows contain duplicate issue/horizon keys")
    grid = _prepare_forecast_grid(forecasts)
    error_history = _weather_error_history(forecasts, analysis, weights)
    if not error_history.empty:
        for column in ("valid_timestamp_utc", "observed_at_utc"):
            error_history[column] = pd.to_datetime(
                error_history[column], utc=True, errors="raise"
            )

    rows: list[dict[str, object]] = []
    for model_row in result.itertuples(index=False):
        issue = model_row.issue_timestamp_utc
        target = model_row.target_timestamp_utc
        at_target = grid.loc[
            grid["valid_timestamp_utc"].eq(target)
            & grid["published_at_utc"].le(issue)
        ].copy()
        if not at_target.empty:
            at_target = at_target.sort_values("published_at_utc").drop_duplicates(
                "region", keep="last"
            )
        selected = weights.merge(at_target, on="region", how="left", validate="one_to_one")
        available = selected["published_at_utc"].notna()
        selected_available = selected.loc[available].copy()
        row: dict[str, object] = {
            "issue_timestamp_utc": issue,
            "target_timestamp_utc": target,
            "forecast_horizon_minutes": int(model_row.forecast_horizon_minutes),
            "weather_forecast_published_at_utc": (
                selected_available["published_at_utc"].max()
                if not selected_available.empty
                else pd.NaT
            ),
            "weather_forecast_run_initialized_at_utc": (
                selected_available["run_initialized_at_utc"].max()
                if not selected_available.empty
                else pd.NaT
            ),
            "weather_forecast_source_available_flag": int(not selected_available.empty),
            "weather_forecast_age_hours": (
                float(
                    (
                        issue - selected_available["published_at_utc"].max()
                    ).total_seconds()
                    / 3600
                )
                if not selected_available.empty
                else np.nan
            ),
            "weather_wind_region_coverage_ratio": float(
                selected.loc[available, "wind_capacity_weight"].sum()
            ),
            "weather_solar_region_coverage_ratio": float(
                selected.loc[available, "solar_capacity_weight"].sum()
            ),
            "weather_wind_weighted_speed_100m_mps": _weighted_mean(
                selected_available, "wind_speed_100m_mps", "wind_capacity_weight"
            ),
            "weather_wind_weighted_u_100m_mps": _weighted_mean(
                selected_available, "wind_u_100m_mps", "wind_capacity_weight"
            ),
            "weather_wind_weighted_v_100m_mps": _weighted_mean(
                selected_available, "wind_v_100m_mps", "wind_capacity_weight"
            ),
            "weather_wind_weighted_gust_10m_mps": _weighted_mean(
                selected_available, "wind_gusts_10m_mps", "wind_capacity_weight"
            ),
            "weather_wind_weighted_gust_excess_mps": _weighted_mean(
                selected_available, "wind_gust_excess_mps", "wind_capacity_weight"
            ),
            "weather_wind_speed_spatial_spread_mps": _weighted_std(
                selected_available, "wind_speed_100m_mps", "wind_capacity_weight"
            ),
            "weather_wind_gust_spatial_spread_mps": _weighted_std(
                selected_available, "wind_gusts_10m_mps", "wind_capacity_weight"
            ),
            "weather_wind_weighted_pressure_tendency_3h_hpa": _weighted_mean(
                selected_available, "pressure_tendency_3h_hpa", "wind_capacity_weight"
            ),
            "weather_solar_weighted_temperature_2m_c": _weighted_mean(
                selected_available, "temperature_2m_c", "solar_capacity_weight"
            ),
            "weather_solar_weighted_cloud_cover_pct": _weighted_mean(
                selected_available, "cloud_cover_pct", "solar_capacity_weight"
            ),
            "weather_solar_weighted_shortwave_radiation_wm2": _weighted_mean(
                selected_available, "shortwave_radiation_wm2", "solar_capacity_weight"
            ),
        }
        for region in weights["region"]:
            match = selected.loc[selected["region"].eq(region)]
            has_value = bool(match["published_at_utc"].notna().any())
            prefix = f"weather_region_{region}"
            row[f"{prefix}_missing_flag"] = int(not has_value)
            for output, source in (
                ("wind_speed_100m_mps", "wind_speed_100m_mps"),
                ("wind_gusts_10m_mps", "wind_gusts_10m_mps"),
                ("temperature_2m_c", "temperature_2m_c"),
                ("cloud_cover_pct", "cloud_cover_pct"),
                ("shortwave_radiation_wm2", "shortwave_radiation_wm2"),
            ):
                row[f"{prefix}_{output}"] = (
                    float(match.iloc[0][source])
                    if has_value and pd.notna(match.iloc[0][source])
                    else np.nan
                )
        row.update(_error_features(error_history, issue))
        rows.append(row)

    features = pd.DataFrame(rows)
    checks = validate_weather_feature_matrix(features)
    checks.update(
        {
            "rows": int(len(features)),
            "columns": int(len(features.columns)),
            "first_issue_utc": features["issue_timestamp_utc"].min().isoformat(),
            "last_issue_utc": features["issue_timestamp_utc"].max().isoformat(),
            "forecast_source_coverage": float(
                features["weather_forecast_source_available_flag"].mean()
            ),
            "mean_wind_region_coverage": float(
                features["weather_wind_region_coverage_ratio"].mean()
            ),
            "mean_solar_region_coverage": float(
                features["weather_solar_region_coverage_ratio"].mean()
            ),
            "error_source_coverage": float(
                features["weather_error_source_available_flag"].mean()
            ),
            "missing_rate_by_feature": {
                column: float(features[column].isna().mean())
                for column in features.columns
                if pd.api.types.is_numeric_dtype(features[column])
            },
        }
    )
    return features, _dictionary(features), checks


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ablation_model() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            ),
            (
                "model",
                HistGradientBoostingRegressor(
                    loss="squared_error",
                    learning_rate=0.05,
                    max_iter=100,
                    max_leaf_nodes=31,
                    min_samples_leaf=10,
                    l2_regularization=1.0,
                    early_stopping=False,
                    random_state=42,
                ),
            ),
        ]
    )


def _score_ablation_model(
    train: pd.DataFrame,
    score: pd.DataFrame,
    columns: list[str],
) -> float:
    model = _ablation_model()
    residual = (
        train["dispatch_down_mwh"]
        - train["dispatch_down_mwh_latest_observed"]
    )
    model.fit(train[columns], residual)
    prediction = np.maximum(
        score["dispatch_down_mwh_latest_observed"].to_numpy(dtype=float)
        + model.predict(score[columns]),
        0.0,
    )
    return float(mean_absolute_error(score["dispatch_down_mwh"], prediction))


def build_weather_ablation_report(
    model_data: pd.DataFrame,
    feature_table: pd.DataFrame,
    feature_artifact_path: Path | str,
) -> dict[str, object]:
    """Run the named weather ablation without touching the final 15% test."""

    keys = [
        "issue_timestamp_utc",
        "target_timestamp_utc",
        "forecast_horizon_minutes",
    ]
    baseline = model_data.copy()
    additions = feature_table.copy()
    for frame in (baseline, additions):
        frame["issue_timestamp_utc"] = pd.to_datetime(
            frame["issue_timestamp_utc"], utc=True, errors="raise"
        )
        frame["target_timestamp_utc"] = pd.to_datetime(
            frame["target_timestamp_utc"], utc=True, errors="raise"
        )
    overlap = baseline.merge(additions, on=keys, how="inner", validate="one_to_one")
    if len(overlap) != len(baseline):
        raise DataQualityError("Weather ablation requires one feature row per baseline row")

    excluded = IDENTIFIER_COLUMNS | TARGET_COLUMNS
    baseline_columns = [
        column
        for column in baseline.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(baseline[column])
    ]
    weather_columns = [
        column
        for column in additions.columns
        if column not in keys
        and pd.api.types.is_numeric_dtype(additions[column])
        and additions[column].notna().any()
    ]
    if not weather_columns:
        raise DataQualityError("No usable regional weather features for ablation")

    unique_times = np.array(sorted(overlap["issue_timestamp_utc"].unique()))
    if len(unique_times) < 20:
        raise DataQualityError("Weather ablation requires at least 20 issue times")
    train_end = int(len(unique_times) * 0.70)
    validation_end = int(len(unique_times) * 0.85)
    train_times = unique_times[:train_end]
    validation_times = unique_times[train_end:validation_end]
    development = overlap.loc[
        overlap["issue_timestamp_utc"].isin(unique_times[:validation_end])
    ].copy()
    folds: list[tuple[str, np.ndarray, np.ndarray]] = []
    for index, (fit_fraction, score_fraction) in enumerate(
        ((0.40, 0.60), (0.60, 0.80), (0.80, 1.00)), start=1
    ):
        fit_end = max(1, int(len(train_times) * fit_fraction))
        score_end = max(fit_end + 1, int(len(train_times) * score_fraction))
        folds.append(
            (
                f"train_rolling_{index}",
                train_times[:fit_end],
                train_times[fit_end:score_end],
            )
        )
    folds.append(("validation", train_times, validation_times))

    candidate_variants = {
        "all_weather": weather_columns,
        "aggregate_forecast_and_errors": [
            column
            for column in weather_columns
            if not column.startswith("weather_region_")
        ],
        "aggregate_forecast_only": [
            column
            for column in weather_columns
            if not column.startswith("weather_region_") and "error" not in column
        ],
        "causal_errors_only": [
            column for column in weather_columns if "error" in column
        ],
        "wind_aggregate_only": [
            column for column in weather_columns if column.startswith("weather_wind_")
        ],
        "solar_aggregate_only": [
            column for column in weather_columns if column.startswith("weather_solar_")
        ],
    }
    # Small synthetic/test tables can make multiple definitions identical. Keep
    # only non-empty, unique feature sets while preserving the declared order.
    unique_variants: dict[str, list[str]] = {}
    seen: set[tuple[str, ...]] = set()
    for variant_name, columns in candidate_variants.items():
        signature = tuple(columns)
        if columns and signature not in seen:
            unique_variants[variant_name] = columns
            seen.add(signature)

    fold_frames: list[tuple[str, pd.DataFrame, pd.DataFrame, float]] = []
    for name, fit_times, score_times in folds:
        fit = development.loc[development["issue_timestamp_utc"].isin(fit_times)]
        score = development.loc[development["issue_timestamp_utc"].isin(score_times)]
        fold_frames.append(
            (
                name,
                fit,
                score,
                _score_ablation_model(fit, score, baseline_columns),
            )
        )

    variant_reports: dict[str, dict[str, object]] = {}
    for variant_name, variant_columns in unique_variants.items():
        fold_metrics: list[dict[str, object]] = []
        for name, fit, score, baseline_mae in fold_frames:
            candidate_mae = _score_ablation_model(
                fit,
                score,
                baseline_columns + variant_columns,
            )
            improvement = (
                (baseline_mae - candidate_mae) / baseline_mae
                if baseline_mae > 0
                else 0.0
            )
            fold_metrics.append(
                {
                    "fold": name,
                    "fit_rows": int(len(fit)),
                    "score_rows": int(len(score)),
                    "baseline_mae_mwh": baseline_mae,
                    "candidate_mae_mwh": candidate_mae,
                    "mae_improvement_fraction": float(improvement),
                }
            )
        improvements = np.array(
            [float(metric["mae_improvement_fraction"]) for metric in fold_metrics]
        )
        variant_reports[variant_name] = {
            "feature_count": len(variant_columns),
            "features": variant_columns,
            "folds": fold_metrics,
            "mean_baseline_mae_mwh": float(
                np.mean([metric["baseline_mae_mwh"] for metric in fold_metrics])
            ),
            "mean_candidate_mae_mwh": float(
                np.mean([metric["candidate_mae_mwh"] for metric in fold_metrics])
            ),
            "mean_mae_improvement_fraction": float(improvements.mean()),
            "worst_fold_improvement_fraction": float(improvements.min()),
        }

    selected_variant = max(
        variant_reports,
        key=lambda name: float(
            variant_reports[name]["mean_mae_improvement_fraction"]
        ),
    )
    selected = variant_reports[selected_variant]
    mean_improvement = float(selected["mean_mae_improvement_fraction"])
    worst_improvement = float(selected["worst_fold_improvement_fraction"])
    selected_columns = unique_variants[selected_variant]
    include = bool(mean_improvement >= 0.15 and worst_improvement >= 0.0)
    artifact = Path(feature_artifact_path)
    return {
        "experiment_name": "regional_weather_signals",
        "feature_group": "capacity_weighted_regional_weather_forecast_and_error",
        "feature_artifact": artifact.as_posix(),
        "feature_artifact_sha256": _sha256(artifact),
        "overlap_rows": int(len(overlap)),
        "development_rows": int(len(development)),
        "development_fold_count": len(folds),
        "evaluated_variant_count": len(variant_reports),
        "selected_variant": selected_variant,
        "candidate_feature_count": len(selected_columns),
        "candidate_features": selected_columns,
        "model": "hist_gradient_boosting_dispatch_residual",
        "selection_gate": {
            "mean_mae_improvement_min": 0.15,
            "maximum_fold_degradation": 0.0,
        },
        "metrics": selected,
        "variants": variant_reports,
        "status": "accepted" if include else "rejected_development_gate",
        "production_decision": "include" if include else "exclude",
        "reason": (
            "Candidate improved mean per-fold MAE by at least 15% without degrading any fold."
            if include
            else "Candidate did not clear the 15% mean per-fold MAE gate with no degrading fold."
        ),
        "final_test_accessed": False,
    }
