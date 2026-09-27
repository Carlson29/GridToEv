"""Complete Swagger request examples, never used as training or live inputs.

V1 source values come from the bundled historical model dataset. That dataset
does not record each publisher's release time, so the example availability
times demonstrate the request contract only; they are not verified vintages.

V2 values are Open-Meteo GFS ``previous_day1`` forecasts for 2024-04-01 at
the four configured coordinates. They were read from the public Previous Runs
API; the availability times follow its target-hour-minus-one-day convention.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .raw_prediction import V1_RAW_CURRENT_FIELDS, V1_RAW_HISTORY_FIELDS


def _utc_iso(value: object) -> str:
    return pd.Timestamp(value).tz_convert("UTC").isoformat().replace("+00:00", "Z")


def v1_raw_example(dataset: pd.DataFrame | None, dataset_path: Path | None) -> dict | None:
    """Return a full historical replay body, if the serving dataset exists."""

    required = {
        "issue_timestamp_utc", "forecast_horizon_minutes",
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
        "forecast_horizon_minutes": 30,
        "flexible_load_capacity_mw": 100,
        "current_observation": current_values,
        "history": history,
    }


# https://previous-runs-api.open-meteo.com/v1/forecast
# Open-Meteo Previous Runs API, gfs_global, 2024-04-01, timezone=UTC,
# wind_speed_unit=kmh, coordinates from daily_curtailment.REGIONS.
# Tuple order: wind speed at 100 m (km/h), shortwave radiation (W/m²),
# temperature at 2 m (°C); each series runs from hour 00 to hour 23 UTC.
_ARCHIVED_FORECASTS = {
    "west": (
        (34.1, 36.4, 37.8, 37.5, 39.1, 38.1, 35.5, 34.5, 33, 31.4, 31.7, 32.7, 34.4, 34.9, 37.4, 37.3, 37.7, 38.9, 22.6, 18.8, 13.1, 10.8, 12.3, 9.9),
        (0, 0, 0, 0, 0, 0, 0, 4, 19, 21, 50, 212, 129, 90, 136, 296, 192, 159, 14, 6, 0, 0, 0, 0),
        (6, 6.3, 6.4, 6.9, 7, 7.1, 6.9, 6.9, 7.1, 7.3, 7.7, 8.5, 7.6, 7.4, 8.2, 9.6, 8.7, 8.3, 6.9, 6.7, 6.6, 6.1, 5.9, 6.1),
    ),
    "south": (
        (27.2, 28.3, 31.4, 35.6, 35.6, 34.1, 37.3, 35.7, 27.7, 29.1, 34.9, 31.4, 12.9, 10.7, 7.7, 3.2, 4.4, 6.1, 6.8, 6.6, 7.3, 7, 6.8, 8.3),
        (0, 0, 0, 0, 0, 0, 0, 3, 16, 59, 388, 537, 79, 58, 97, 107, 97, 52, 138, 38, 0, 0, 0, 0),
        (7.6, 7.8, 8, 7.9, 7.9, 7.5, 7.5, 7.2, 7.4, 8.2, 10, 11.1, 9.4, 9.4, 9.6, 9.4, 9.2, 8.9, 9.8, 7.6, 7.1, 7.8, 7.8, 7.7),
    ),
    "east": (
        (34.9, 30.3, 32.8, 30.9, 25.2, 20.7, 14.7, 11, 11.7, 12.1, 14.5, 15.7, 19.4, 22.9, 21.3, 19.5, 17.4, 15.1, 18.2, 17.6, 16.2, 14.3, 15.1, 15.4),
        (0, 0, 0, 0, 0, 0, 0, 3, 69, 225, 414, 524, 259, 445, 459, 337, 431, 295, 11, 2, 0, 0, 0, 0),
        (8.4, 8.2, 8, 7.8, 7.8, 7.8, 8, 8, 8.5, 8.9, 9.5, 9.9, 9.1, 9.3, 9.6, 9.5, 9.8, 9.2, 6.8, 6.8, 6.8, 6.8, 6.7, 6.2),
    ),
    "north": (
        (42, 40, 42.3, 39.2, 37.6, 38.7, 40.9, 42.6, 42.2, 47.8, 47.2, 51.6, 42.9, 42.9, 40.2, 35.4, 29.9, 28, 26.7, 25.7, 22.9, 20.1, 17.6, 16.6),
        (0, 0, 0, 0, 0, 0, 0, 2, 11, 16, 23, 84, 237, 335, 214, 238, 117, 67, 11, 4, 0, 0, 0, 0),
        (6.3, 6.1, 6.7, 6, 5.9, 5.7, 5.6, 5.7, 5.6, 5.5, 5.6, 6.3, 7.3, 7.9, 7.7, 8.1, 7.7, 7.4, 6.8, 6, 6, 6, 6, 6),
    ),
}


def v2_raw_example() -> dict:
    """Return 96 archived source forecasts with coherent target/vintage times."""

    records = []
    for region, (wind, solar, temperature) in _ARCHIVED_FORECASTS.items():
        for hour in range(24):
            records.append({
                "region": region,
                "target_hour_utc": f"2024-04-01T{hour:02d}:00:00Z",
                "forecast_available_at_utc": f"2024-03-31T{hour:02d}:00:00Z",
                "wind_speed_100m_kmh": wind[hour],
                "shortwave_radiation_wm2": solar[hour],
                "temperature_2m_c": temperature[hour],
            })
    return {"target_date_utc": "2024-04-01", "hourly_forecasts": records}
