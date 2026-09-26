"""Leakage-conscious, UTC-day curtailment data built from archived forecasts.

The Open-Meteo ``previous_day1`` variables are forecasts for a valid hour
made 24 hours before that hour. At 00:00 UTC on D, all D hourly values are
therefore already forecast. EirGrid dispatch-down is used only as the label.
"""

from __future__ import annotations

import json
import time
from datetime import date, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

from .constants import PROJECT_ROOT


FORECAST_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
FORECAST_MODEL = "gfs_global"
REGIONS = {
    "west": (53.27, -9.05),  # Galway / Atlantic wind
    "south": (51.90, -8.47),  # Cork
    "east": (53.35, -6.26),  # Dublin
    "north": (54.60, -5.93),  # Belfast / Northern Ireland
}
API_VARIABLES = (
    "wind_speed_100m_previous_day1",
    "shortwave_radiation_previous_day1",
    "temperature_2m_previous_day1",
)
VALUE_COLUMNS = (
    "wind_speed_100m_kmh",
    "shortwave_radiation_wm2",
    "temperature_2m_c",
)
DEFAULT_HISTORY = PROJECT_ROOT / "data" / "processed" / "eirgrid_core_history_30min.csv.gz"
DEFAULT_DAILY_DATASET = PROJECT_ROOT / "data" / "processed" / "daily_curtailment_forecast_v2.csv.gz"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "open_meteo_previous_runs"


def _request_json(params: dict[str, object], *, attempts: int = 3) -> dict:
    url = f"{FORECAST_URL}?{urlencode(params)}"
    for attempt in range(attempts):
        try:
            with urlopen(url, timeout=45) as response:
                payload = json.load(response)
            if payload.get("error"):
                raise ValueError(f"Open-Meteo rejected request: {payload.get('reason')}")
            return payload
        except HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == attempts - 1:
                raise
        except (URLError, TimeoutError):
            if attempt == attempts - 1:
                raise
        time.sleep(2**attempt)
    raise RuntimeError("Unreachable forecast download state")


def fetch_previous_runs(
    region: str,
    start: date,
    end: date,
    *,
    raw_dir: Path = DEFAULT_RAW_DIR,
    refresh: bool = False,
    cache: bool = True,
) -> pd.DataFrame:
    """Cache source JSON in git-ignored raw data; never silently use stale partials."""

    if region not in REGIONS:
        raise ValueError(f"Unknown region: {region}")
    if end < start:
        raise ValueError("end must not precede start")
    path = raw_dir / FORECAST_MODEL / region / f"{start.isoformat()}_{end.isoformat()}.json"
    if cache and path.exists() and not refresh:
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        latitude, longitude = REGIONS[region]
        payload = _request_json({
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "hourly": ",".join(API_VARIABLES),
            "models": FORECAST_MODEL,
            "timezone": "UTC",
            "temperature_unit": "celsius",
            "wind_speed_unit": "kmh",
        })
        if cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return parse_previous_runs(payload, region)


def parse_previous_runs(payload: dict, region: str) -> pd.DataFrame:
    if payload.get("utc_offset_seconds") != 0:
        raise ValueError("Forecast response must use UTC")
    hourly = payload.get("hourly", {})
    missing = (set(API_VARIABLES) | {"time"}) - set(hourly)
    if missing:
        raise ValueError(f"Forecast response missing {sorted(missing)}")
    lengths = {len(hourly[key]) for key in ("time", *API_VARIABLES)}
    if len(lengths) != 1:
        raise ValueError("Forecast response arrays differ in length")
    target_hours = pd.to_datetime(hourly["time"], utc=True, errors="raise")
    if target_hours.has_duplicates:
        raise ValueError("Forecast response contains duplicate target hours")
    frame = pd.DataFrame({
        "target_hour_utc": target_hours,
        "region": region,
        "available_at_utc": target_hours - pd.Timedelta(days=1),
    })
    for source, destination in zip(API_VARIABLES, VALUE_COLUMNS):
        frame[destination] = pd.to_numeric(hourly[source], errors="coerce")
    return frame


def build_forecast_features(
    forecasts: pd.DataFrame,
    *,
    regions: tuple[str, ...] = tuple(REGIONS),
) -> pd.DataFrame:
    """Keep only days with all 24 valid hours for every requested region."""

    if not regions or len(set(regions)) != len(regions):
        raise ValueError("regions must be a nonempty unique sequence")
    if forecasts.empty:
        return pd.DataFrame()
    required = {"target_hour_utc", "region", "available_at_utc", *VALUE_COLUMNS}
    if required - set(forecasts):
        raise ValueError(f"Missing forecast columns: {sorted(required - set(forecasts))}")
    data = forecasts.copy()
    data["target_hour_utc"] = pd.to_datetime(data["target_hour_utc"], utc=True)
    data["available_at_utc"] = pd.to_datetime(data["available_at_utc"], utc=True)
    data["issue_timestamp_utc"] = data["target_hour_utc"].dt.floor("D")
    if data.duplicated(["region", "target_hour_utc"]).any():
        raise ValueError("Duplicate region/target-hour forecast")
    results = []
    for (day, region), group in data.groupby(["issue_timestamp_utc", "region"], sort=True):
        if region not in regions:
            continue
        expected = pd.date_range(day, periods=24, freq="h", tz="UTC")
        if len(group) != 24 or not group["target_hour_utc"].sort_values().reset_index(drop=True).equals(pd.Series(expected)):
            continue
        if group[list(VALUE_COLUMNS)].isna().any().any():
            continue
        if not (group["available_at_utc"] <= day).all():
            continue
        values = {
            "issue_timestamp_utc": day,
            "region": region,
            "forecast_max_available_at_utc": group["available_at_utc"].max(),
            "wind_mean_kmh": group["wind_speed_100m_kmh"].mean(),
            "wind_max_kmh": group["wind_speed_100m_kmh"].max(),
            "wind_p75_kmh": group["wind_speed_100m_kmh"].quantile(0.75),
            "solar_sum_wm2h": group["shortwave_radiation_wm2"].clip(lower=0).sum(),
            "temperature_mean_c": group["temperature_2m_c"].mean(),
        }
        results.append(values)
    summary = pd.DataFrame(results)
    if summary.empty:
        return pd.DataFrame()
    counts = summary.groupby("issue_timestamp_utc")["region"].nunique()
    full_days = counts[counts.eq(len(regions))].index
    summary = summary.loc[summary["issue_timestamp_utc"].isin(full_days)]
    wide = summary.pivot(index="issue_timestamp_utc", columns="region")
    wide.columns = [f"{region}_{metric}" for metric, region in wide.columns]
    wide = wide.reset_index()
    available = [f"{region}_forecast_max_available_at_utc" for region in regions]
    wide["forecast_max_available_at_utc"] = wide[available].max(axis=1)
    wide = wide.drop(columns=available)
    wind = [f"{region}_wind_mean_kmh" for region in regions]
    wide["wind_regional_spread_kmh"] = wide[wind].max(axis=1) - wide[wind].min(axis=1)
    day_of_year = wide["issue_timestamp_utc"].dt.dayofyear
    wide["calendar_day_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    wide["calendar_day_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    wide["calendar_weekend"] = wide["issue_timestamp_utc"].dt.dayofweek.ge(5).astype(int)
    return wide.sort_values("issue_timestamp_utc").reset_index(drop=True)


def build_daily_labels(history: pd.DataFrame) -> pd.DataFrame:
    """EirGrid MWh is already energy per half-hour; sum 48 complete UTC rows."""

    required = {"timestamp_utc", "curtailment_mwh", "dispatch_down_available_flag"}
    if required - set(history):
        raise ValueError(f"Missing EirGrid label columns: {sorted(required - set(history))}")
    data = history[list(required)].copy()
    data["timestamp_utc"] = pd.to_datetime(data["timestamp_utc"], utc=True)
    data["curtailment_mwh"] = pd.to_numeric(data["curtailment_mwh"], errors="coerce")
    data["issue_timestamp_utc"] = data["timestamp_utc"].dt.floor("D")
    if data["timestamp_utc"].duplicated().any():
        raise ValueError("Duplicate EirGrid label timestamps")
    results = []
    for day, group in data.groupby("issue_timestamp_utc", sort=True):
        expected = pd.date_range(day, periods=48, freq="30min", tz="UTC")
        if len(group) != 48 or not group["timestamp_utc"].sort_values().reset_index(drop=True).equals(pd.Series(expected)):
            continue
        if not group["dispatch_down_available_flag"].eq(1).all():
            continue
        if group["curtailment_mwh"].isna().any() or group["curtailment_mwh"].lt(0).any():
            continue
        amount = float(group["curtailment_mwh"].sum())
        results.append({
            "issue_timestamp_utc": day,
            "curtailment_mwh": amount,
            "curtailment_event": int(amount > 0),
        })
    return pd.DataFrame(results)


def date_chunks(start: date, end: date, days: int = 60):
    cursor = start
    while cursor <= end:
        stop = min(cursor + timedelta(days=days - 1), end)
        yield cursor, stop
        cursor = stop + timedelta(days=1)


def build_dataset(
    *,
    start: date,
    end: date,
    history_path: Path = DEFAULT_HISTORY,
    raw_dir: Path = DEFAULT_RAW_DIR,
    regions: tuple[str, ...] = tuple(REGIONS),
    refresh: bool = False,
) -> pd.DataFrame:
    forecasts = pd.concat(
        [fetch_previous_runs(region, first, last, raw_dir=raw_dir, refresh=refresh)
         for region in regions for first, last in date_chunks(start, end)],
        ignore_index=True,
    )
    features = build_forecast_features(forecasts, regions=regions)
    history = pd.read_csv(
        history_path,
        usecols=["timestamp_utc", "curtailment_mwh", "dispatch_down_available_flag"],
    )
    labels = build_daily_labels(history)
    if features.empty or labels.empty:
        raise ValueError("No overlapping complete forecast/label days")
    result = features.merge(labels, on="issue_timestamp_utc", how="inner", validate="one_to_one")
    if result.empty:
        raise ValueError("No overlapping complete forecast/label days")
    return result.sort_values("issue_timestamp_utc").reset_index(drop=True)
