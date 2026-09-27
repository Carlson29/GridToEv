"""Deterministic source-value adapters for the two frozen model feature contracts.

These adapters do not collect data or assert that user-supplied publication
timestamps are genuine. They reject missing history and post-issue values.
"""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd

from .daily_curtailment import REGIONS, build_forecast_features


V1_GENERATION_FIELDS = (
    "entsoe_fossil_gas_generation_mw",
    "entsoe_fossil_hard_coal_generation_mw",
    "entsoe_fossil_oil_generation_mw",
    "entsoe_fossil_peat_generation_mw",
    "entsoe_hydro_pumped_storage_generation_mw",
    "entsoe_hydro_run_of_river_and_poundage_generation_mw",
    "entsoe_other_generation_mw",
    "entsoe_solar_generation_mw",
    "entsoe_wind_onshore_generation_mw",
)
V1_EIRGRID_FIELDS = (
    "eirgrid_ie_generation_mw",
    "eirgrid_ie_demand_mw",
    "eirgrid_ie_wind_availability_mw",
    "eirgrid_ie_wind_generation_mw",
    "eirgrid_ie_solar_availability_mw",
    "eirgrid_ie_solar_generation_mw",
    "eirgrid_ie_hydro_generation_mw",
    "eirgrid_ewic_flow_mw",
    "eirgrid_greenlink_flow_mw",
    "eirgrid_ie_wind_penetration_ratio",
    "eirgrid_ie_solar_penetration_ratio",
    "eirgrid_all_island_generation_mw",
    "eirgrid_all_island_demand_mw",
    "eirgrid_all_island_wind_availability_mw",
    "eirgrid_all_island_wind_generation_mw",
    "eirgrid_all_island_solar_availability_mw",
    "eirgrid_all_island_solar_generation_mw",
    "eirgrid_all_island_hydro_generation_mw",
    "eirgrid_interjurisdictional_flow_mw",
    "eirgrid_all_island_wind_penetration_ratio",
    "eirgrid_all_island_solar_penetration_ratio",
    "eirgrid_all_island_oversupply_mw",
    "eirgrid_all_island_oversupply_ratio",
    "eirgrid_snsp_ratio",
)
V1_RAW_CURRENT_FIELDS = (
    *V1_GENERATION_FIELDS,
    "entsoe_actual_load_mw",
    "entsoe_price_eur_mwh",
    *V1_EIRGRID_FIELDS,
)
V1_RAW_HISTORY_FIELDS = (
    "eirgrid_ie_wind_generation_mw",
    "eirgrid_ie_demand_mw",
    "entsoe_price_eur_mwh",
    "eirgrid_snsp_ratio",
    "eirgrid_all_island_oversupply_mw",
)
V1_RATIO_FIELDS = frozenset(name for name in V1_EIRGRID_FIELDS if name.endswith("_ratio"))
V1_SIGNED_FIELDS = frozenset({
    "entsoe_price_eur_mwh",
    "eirgrid_ewic_flow_mw",
    "eirgrid_greenlink_flow_mw",
    "eirgrid_interjurisdictional_flow_mw",
})
V1_HISTORY_STEPS = 48


def _utc_timestamp(value: object, label: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a timezone-aware UTC timestamp") from error
    if timestamp.tzinfo is None:
        raise ValueError(f"{label} must include a timezone")
    return timestamp.tz_convert("UTC")


def _numeric(value: object, label: str, *, signed: bool = False, ratio: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not np.isfinite(number):
        raise ValueError(f"{label} must be finite")
    if not signed and number < 0:
        raise ValueError(f"{label} must be nonnegative")
    if ratio and number > 1:
        raise ValueError(f"{label} must be a 0–1 fraction, not a 0–100 percentage")
    return number


def build_v1_features_from_raw(
    issue_timestamp_utc: str | datetime | pd.Timestamp,
    current: dict,
    history: list[dict],
    forecast_horizon_minutes: int,
) -> dict[str, float]:
    """Recreate the final V1 model row from a complete source snapshot/history.

    The notebook's source interpolation is intentionally not repeated here:
    this path requires every raw value, so its three imputation flags are zero.
    """

    issue = _utc_timestamp(issue_timestamp_utc, "issue_timestamp_utc")
    if issue.second or issue.microsecond or issue.minute not in (0, 30):
        raise ValueError("issue_timestamp_utc must align to a UTC half-hour")
    if forecast_horizon_minutes not in (30, 60):
        raise ValueError("forecast_horizon_minutes must be 30 or 60")
    if len(history) != V1_HISTORY_STEPS:
        raise ValueError("Exactly 48 consecutive prior half-hour observations are required")
    if _utc_timestamp(current.get("available_at_utc"), "current.available_at_utc") > issue:
        raise ValueError("Current raw values were not available at issue time")

    required_current = set(V1_RAW_CURRENT_FIELDS) | {"observed_dispatch_down_mwh", "available_at_utc"}
    if required_current - set(current):
        raise ValueError(f"Missing current raw values: {sorted(required_current - set(current))}")
    required_history = set(V1_RAW_HISTORY_FIELDS) | {
        "timestamp_utc", "available_at_utc", "observed_dispatch_down_mwh",
    }
    expected_times = pd.date_range(
        issue - pd.Timedelta(minutes=30 * V1_HISTORY_STEPS),
        periods=V1_HISTORY_STEPS, freq="30min", tz="UTC",
    )
    for expected, record in zip(expected_times, history):
        if required_history - set(record):
            raise ValueError(f"Missing history raw values: {sorted(required_history - set(record))}")
        if _utc_timestamp(record["timestamp_utc"], "history.timestamp_utc") != expected:
            raise ValueError("Exactly 48 consecutive prior half-hour observations are required")
        if _utc_timestamp(record["available_at_utc"], "history.available_at_utc") > issue:
            raise ValueError("A history value was not available at issue time")
    for index in range(1, len(history)):
        hour = expected_times[index]
        if hour.minute == 30 and not np.isclose(
            _numeric(history[index]["entsoe_price_eur_mwh"], "history.entsoe_price_eur_mwh", signed=True),
            _numeric(history[index - 1]["entsoe_price_eur_mwh"], "history.entsoe_price_eur_mwh", signed=True),
        ):
            raise ValueError("At :30 each historical ENTSO-E price must equal the preceding :00 price")

    features = {
        name: _numeric(
            current[name], name,
            signed=name in V1_SIGNED_FIELDS, ratio=name in V1_RATIO_FIELDS,
        )
        for name in V1_RAW_CURRENT_FIELDS
    }
    if features["eirgrid_ie_demand_mw"] <= 0:
        raise ValueError("eirgrid_ie_demand_mw must be positive for ratio features")
    features["entsoe_generation_imputed_flag"] = 0
    features["entsoe_load_imputed_flag"] = 0
    features["eirgrid_system_imputed_flag"] = 0
    features["entsoe_price_hourly_carry_forward_flag"] = int(issue.minute == 30)
    if issue.minute == 30 and not np.isclose(
        features["entsoe_price_eur_mwh"],
        _numeric(history[-1]["entsoe_price_eur_mwh"], "history.entsoe_price_eur_mwh", signed=True),
    ):
        raise ValueError("At :30 the ENTSO-E hourly price must equal the preceding :00 price")

    features["entsoe_total_generation_mw"] = sum(features[name] for name in V1_GENERATION_FIELDS)
    features["entsoe_renewable_generation_mw"] = sum(features[name] for name in (
        "entsoe_hydro_run_of_river_and_poundage_generation_mw",
        "entsoe_solar_generation_mw", "entsoe_wind_onshore_generation_mw",
    ))
    features["entsoe_nonrenewable_generation_mw"] = (
        features["entsoe_total_generation_mw"] - features["entsoe_renewable_generation_mw"]
    )
    wind = features["eirgrid_ie_wind_generation_mw"]
    solar = features["eirgrid_ie_solar_generation_mw"]
    demand = features["eirgrid_ie_demand_mw"]
    features["eirgrid_ie_variable_renewable_generation_mw"] = wind + solar
    features["eirgrid_ie_variable_renewable_availability_mw"] = (
        features["eirgrid_ie_wind_availability_mw"] + features["eirgrid_ie_solar_availability_mw"]
    )
    features["eirgrid_ie_renewable_headroom_mw"] = max(
        features["eirgrid_ie_variable_renewable_availability_mw"] - wind - solar, 0.0,
    )
    features["eirgrid_ie_net_load_mw"] = demand - wind - solar
    features["eirgrid_ie_renewable_share_ratio"] = (wind + solar) / demand
    features["eirgrid_snsp_headroom_to_75pct"] = 0.75 - features["eirgrid_snsp_ratio"]
    features["entsoe_price_negative_flag"] = int(features["entsoe_price_eur_mwh"] < 0)
    features["entsoe_price_below_20_flag"] = int(features["entsoe_price_eur_mwh"] < 20)
    features["high_wind_low_load_interaction"] = wind / demand

    half_hour_slot = issue.hour * 2 + issue.minute // 30
    features["hour_sin"] = float(np.sin(2 * np.pi * half_hour_slot / 48))
    features["hour_cos"] = float(np.cos(2 * np.pi * half_hour_slot / 48))
    features["weekday"] = issue.dayofweek
    features["weekday_sin"] = float(np.sin(2 * np.pi * issue.dayofweek / 7))
    features["weekday_cos"] = float(np.cos(2 * np.pi * issue.dayofweek / 7))
    features["is_weekend"] = int(issue.dayofweek >= 5)
    features["month"] = issue.month

    latest_dispatch = _numeric(
        current["observed_dispatch_down_mwh"], "current.observed_dispatch_down_mwh",
    )
    features["dispatch_down_mwh_latest_observed"] = latest_dispatch
    features["dispatch_down_event_latest_observed"] = int(latest_dispatch > 0)
    sources = {
        "wind": "eirgrid_ie_wind_generation_mw",
        "load": "eirgrid_ie_demand_mw",
        "price": "entsoe_price_eur_mwh",
        "snsp": "eirgrid_snsp_ratio",
        "oversupply": "eirgrid_all_island_oversupply_mw",
    }
    for short_name, source in sources.items():
        values = [
            _numeric(
                record[source], f"history.{source}",
                signed=source in V1_SIGNED_FIELDS, ratio=source in V1_RATIO_FIELDS,
            )
            for record in history
        ] + [features[source]]
        for lag in (1, 2, 4, 12, 48):
            features[f"{short_name}_lag_{lag}"] = values[-1 - lag]
        features[f"{short_name}_ramp_30m"] = values[-1] - values[-2]
        features[f"{short_name}_ramp_60m"] = values[-1] - values[-3]
        if short_name in ("wind", "load", "price"):
            for window in (6, 12, 48):
                past = np.asarray(values[-1 - window:-1], dtype=float)
                features[f"{short_name}_rolling_mean_{window}"] = float(past.mean())
                if short_name in ("wind", "price"):
                    features[f"{short_name}_rolling_std_{window}"] = float(past.std(ddof=1))

    dispatch_history = [
        _numeric(record["observed_dispatch_down_mwh"], "history.observed_dispatch_down_mwh")
        for record in history
    ] + [latest_dispatch]
    for lag in (1, 2, 4, 48):
        previous = dispatch_history[-1 - lag]
        features[f"dispatch_down_mwh_lag_{lag}"] = previous
        features[f"dispatch_down_event_lag_{lag}"] = int(previous > 0)
    features["forecast_horizon_minutes"] = forecast_horizon_minutes
    return features


def build_v2_features_from_raw(
    target_date_utc: date,
    hourly_forecasts: list[dict],
    issued_at_utc: datetime | pd.Timestamp,
) -> pd.DataFrame:
    """Aggregate 96 complete region-hours exactly as the trained V2 data builder."""

    issued_at = _utc_timestamp(issued_at_utc, "issued_at_utc")
    if len(hourly_forecasts) != 24 * len(REGIONS):
        raise ValueError("A complete UTC day needs 24 forecast hours for each of four regions")
    rows = []
    expected_hours = set(pd.date_range(target_date_utc, periods=24, freq="h", tz="UTC"))
    for record in hourly_forecasts:
        region = record.get("region")
        if region not in REGIONS:
            raise ValueError(f"Unknown forecast region: {region}")
        hour = _utc_timestamp(record.get("target_hour_utc"), "target_hour_utc")
        available = _utc_timestamp(record.get("forecast_available_at_utc"), "forecast_available_at_utc")
        if hour not in expected_hours:
            raise ValueError("Forecast target hour must be within the requested complete UTC day")
        if available > issued_at:
            raise ValueError("A forecast was not available at issue time")
        rows.append({
            "target_hour_utc": hour,
            "region": region,
            "available_at_utc": available,
            "wind_speed_100m_kmh": _numeric(record.get("wind_speed_100m_kmh"), "wind_speed_100m_kmh"),
            "shortwave_radiation_wm2": _numeric(record.get("shortwave_radiation_wm2"), "shortwave_radiation_wm2"),
            "temperature_2m_c": _numeric(record.get("temperature_2m_c"), "temperature_2m_c", signed=True),
        })
    frame = pd.DataFrame(rows)
    if frame.duplicated(["region", "target_hour_utc"]).any():
        raise ValueError("Forecast region/hour values must be unique and complete")
    features = build_forecast_features(frame)
    if len(features) != 1 or features.iloc[0]["issue_timestamp_utc"].date() != target_date_utc:
        raise ValueError("A complete forecast for all four regions is required and must be available by the target day")
    return features
