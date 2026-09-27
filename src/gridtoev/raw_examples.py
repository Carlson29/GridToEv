"""One-step-beyond-dataset Swagger demos, never added to model data.

V1 source values come from the bundled historical model dataset. That dataset
does not record each publisher's release time, so the example availability
times demonstrate the request contract only; they are not verified vintages.

V2 values are Open-Meteo GFS ``previous_day1`` forecasts for 2026-08-31 at
the four configured coordinates. They were read from the public Previous Runs
API; the availability times follow its target-hour-minus-one-day convention.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .raw_prediction import V1_RAW_CURRENT_FIELDS, V1_RAW_HISTORY_FIELDS


def _utc_iso(value: object) -> str:
    return pd.Timestamp(value).tz_convert("UTC").isoformat().replace("+00:00", "Z")


def v1_raw_example(dataset: pd.DataFrame | None, dataset_path: Path | None) -> dict | None:
    """Return the next unlabelled half-hour's raw inputs, when available."""

    required = {
        "issue_timestamp_utc", "target_timestamp_utc", "forecast_horizon_minutes",
        "dispatch_down_mwh_latest_observed", *V1_RAW_CURRENT_FIELDS,
    }
    if dataset is None:
        if dataset_path is None or not dataset_path.is_file():
            return None
        dataset = pd.read_csv(dataset_path)
    if not required.issubset(dataset.columns):
        return None
    rows = (
        dataset.loc[dataset["forecast_horizon_minutes"].eq(30)]
        .sort_values("issue_timestamp_utc")
        .tail(49)
        .reset_index(drop=True)
    )
    if len(rows) != 49:
        return None
    current = rows.iloc[-1]
    issue = _utc_iso(current["issue_timestamp_utc"])
    if pd.Timestamp(issue) != pd.to_datetime(dataset["issue_timestamp_utc"], utc=True).max():
        return None
    # Predict the first half-hour after the final labelled dataset target.
    # For the frozen release dataset this chooses 60 minutes from 22:30 UTC.
    last_target = pd.to_datetime(dataset["target_timestamp_utc"], utc=True).max()
    next_target = last_target + pd.Timedelta(minutes=30)
    lead = next_target - pd.Timestamp(issue)
    if lead not in (pd.Timedelta(minutes=30), pd.Timedelta(minutes=60)):
        return None
    horizon_minutes = int(lead / pd.Timedelta(minutes=1))
    current_values = {name: float(current[name]) for name in V1_RAW_CURRENT_FIELDS}
    current_values.update({
        "observed_dispatch_down_mwh": float(current["dispatch_down_mwh_latest_observed"]),
        "available_at_utc": issue,
    })
    history = []
    for _, row in rows.iloc[:-1].iterrows():
        record = {name: float(row[name]) for name in V1_RAW_HISTORY_FIELDS}
        record.update({
            "timestamp_utc": _utc_iso(row["issue_timestamp_utc"]),
            "available_at_utc": _utc_iso(row["issue_timestamp_utc"]),
            "observed_dispatch_down_mwh": float(row["dispatch_down_mwh_latest_observed"]),
        })
        history.append(record)
    return {
        "issue_timestamp_utc": issue,
        "forecast_horizon_minutes": horizon_minutes,
        "flexible_load_capacity_mw": 100,
        "current_observation": current_values,
        "history": history,
    }


# https://previous-runs-api.open-meteo.com/v1/forecast
# Open-Meteo Previous Runs API, gfs_global, 2026-08-31, timezone=UTC,
# wind_speed_unit=kmh, coordinates from daily_curtailment.REGIONS.
# Tuple order: wind speed at 100 m (km/h), shortwave radiation (W/m²),
# temperature at 2 m (°C); each series runs from hour 00 to hour 23 UTC.
_ARCHIVED_FORECASTS = {
    "west": (
        (38.7, 39, 38.3, 34.8, 30.2, 25.5, 25.8, 23.5, 22.9, 23.1, 25.4, 28.3, 27.5, 26.3, 26.5, 28.3, 32.8, 31.9, 26.7, 27.5, 27, 25.7, 27.2, 27),
        (0, 0, 0, 0, 0, 0, 0, 15, 136, 245, 386, 345, 292, 325, 422, 422, 227, 274, 54, 17, 4, 0, 0, 0),
        (14, 13.3, 13.8, 13.8, 13.7, 13.4, 13, 13, 13.6, 14.7, 15.8, 16.2, 15.9, 16.2, 16.6, 17, 16.4, 15.9, 14.9, 14.9, 14, 13.5, 13.7, 14.1),
    ),
    "south": (
        (41.4, 40.5, 37, 33.4, 31.2, 32, 30.9, 29.4, 24.6, 23.7, 25.1, 25, 27.2, 25.9, 22.9, 21.1, 18.3, 20.2, 18.3, 19.5, 23.9, 23.9, 22.1, 21.6),
        (0, 0, 0, 0, 0, 0, 0, 31, 157, 272, 378, 531, 660, 515, 424, 483, 219, 282, 189, 78, 2, 0, 0, 0),
        (14.1, 13.7, 13.6, 13.2, 13, 13.2, 13.2, 13.7, 14.3, 15.2, 16.2, 17.5, 18.4, 18.4, 18.4, 18.7, 17.8, 17.8, 17.1, 14.4, 12.4, 12.1, 11.9, 11.6),
    ),
    "east": (
        (23, 23.5, 24.9, 26.9, 30.6, 32.9, 33.3, 31.4, 32.4, 28.8, 31.8, 31.6, 31.5, 32.9, 32.6, 30, 31.1, 25.2, 23, 24.6, 22.2, 21.1, 24.5, 25.9),
        (0, 0, 0, 0, 0, 0, 1, 11, 83, 107, 137, 173, 165, 219, 477, 378, 394, 255, 121, 33, 1, 0, 0, 0),
        (13.5, 13.3, 13, 13, 13.4, 13, 13, 13.1, 13.7, 14.1, 14.4, 15, 16.1, 16.4, 17.6, 17.5, 17.5, 17.1, 16.9, 16.2, 15.4, 14.9, 14.3, 14.1),
    ),
    "north": (
        (26.5, 26.1, 28.2, 33.6, 40.1, 41.6, 40.3, 40.5, 39.4, 40.8, 40.2, 42.9, 39.6, 37.6, 32.7, 31.3, 28.2, 23.4, 24.5, 21.1, 21, 26.5, 30.7, 32.9),
        (0, 0, 0, 0, 0, 0, 1, 29, 100, 261, 358, 550, 439, 478, 326, 366, 348, 244, 91, 25, 0, 0, 0, 0),
        (12.2, 12.4, 12.6, 13, 12.5, 12.2, 12.9, 12.7, 12.9, 13.7, 14.4, 15.4, 15.4, 15.8, 15.6, 16, 16.1, 16, 15.3, 14.6, 13.9, 14.3, 13, 12.7),
    ),
}


_V2_EXAMPLE_DATE = date(2026, 8, 31)


def v2_raw_example(report_path: Path) -> dict | None:
    """Return archived forecasts for the first day after V2 model coverage."""

    if not report_path.is_file():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    last_model_date = max(date.fromisoformat(bounds[1][:10]) for bounds in report["date_ranges"].values())
    target_date = last_model_date + timedelta(days=1)
    if target_date != _V2_EXAMPLE_DATE:
        # A changed dataset needs a newly sourced forecast fixture.
        return None

    records = []
    previous_date = target_date - timedelta(days=1)
    for region, (wind, solar, temperature) in _ARCHIVED_FORECASTS.items():
        for hour in range(24):
            records.append({
                "region": region,
                "target_hour_utc": f"{target_date.isoformat()}T{hour:02d}:00:00Z",
                "forecast_available_at_utc": f"{previous_date.isoformat()}T{hour:02d}:00:00Z",
                "wind_speed_100m_kmh": wind[hour],
                "shortwave_radiation_wm2": solar[hour],
                "temperature_2m_c": temperature[hour],
            })
    return {"target_date_utc": target_date.isoformat(), "hourly_forecasts": records}
