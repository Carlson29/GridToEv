"""Read-only observed Wind/Solar curtailment lookups for one UTC day.

These are EirGrid's recorded outcomes from the audited source-label archive,
not model predictions. An unpublished source is reported as unknown (null),
never as zero.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import PROJECT_ROOT
from .source_curtailment import HALF_HOURS_PER_DAY, build_daily_source_labels


DEFAULT_SOURCE_HISTORY = PROJECT_ROOT / "data" / "processed" / "eirgrid_source_curtailment_30min.csv.gz"
_COLUMNS = [
    "timestamp_utc", "wind_curtailment_mwh", "solar_curtailment_mwh",
    "wind_reported_flag", "solar_reported_flag", "source_labels_complete_flag",
]


def _mwh(value: float) -> float:
    """EirGrid publishes three decimals; drop float summation noise."""
    return round(float(value), 3)


def _percent(part: float, total: float) -> float | None:
    return round(100 * part / total, 2) if total > 0 else None


class SourceActualsService:
    def __init__(self, dataset_path: Path | str = DEFAULT_SOURCE_HISTORY) -> None:
        self.dataset_path = Path(dataset_path)
        self._data: pd.DataFrame | None = None
        self._days: pd.DataFrame | None = None

    def _load(self) -> None:
        if self._data is not None:
            return
        data = pd.read_csv(self.dataset_path, usecols=_COLUMNS)
        data["timestamp_utc"] = pd.to_datetime(data["timestamp_utc"], utc=True, errors="raise")
        if data.empty or data["timestamp_utc"].duplicated().any():
            raise ValueError("Source actuals archive is empty or has duplicate UTC half-hours")
        values = data[["wind_curtailment_mwh", "solar_curtailment_mwh"]].to_numpy(dtype=float)
        if (values[np.isfinite(values)] < 0).any():
            raise ValueError("Source actuals archive has negative curtailment")
        self._days = build_daily_source_labels(data).set_index("target_date_utc")
        self._data = data.set_index("timestamp_utc", drop=False).sort_index()

    def coverage(self) -> dict[str, Any]:
        self._load()
        assert self._data is not None and self._days is not None
        solar = self._data.loc[self._data["solar_reported_flag"].eq(1), "timestamp_utc"]
        return {
            "source": "EirGrid half-hourly dispatch-down workbooks, Ireland (IE), Wind and Solar rows",
            "unit": "MWh",
            "first_half_hour_utc": self._data.index.min().isoformat(),
            "last_half_hour_utc": self._data.index.max().isoformat(),
            "solar_first_published_utc": solar.min().isoformat() if len(solar) else None,
            "complete_day_min_utc": self._days.index.min().date().isoformat() if len(self._days) else None,
            "complete_day_max_utc": self._days.index.max().date().isoformat() if len(self._days) else None,
            "complete_day_count": int(len(self._days)),
            "notice": (
                "Recorded outcomes only, not predictions. Before solar was first published, days return "
                "wind only with solar unknown. Later days are pending until the archive is refreshed."
            ),
        }

    def day(self, target_date: date, *, include_half_hours: bool = False) -> dict[str, Any]:
        self._load()
        assert self._data is not None and self._days is not None
        start = pd.Timestamp(target_date, tz="UTC")
        rows = self._data.loc[start: start + pd.Timedelta(hours=23, minutes=30)]
        latest = self._data.index.max()
        result: dict[str, Any] = {
            "status": "pending" if start + pd.Timedelta(hours=23, minutes=30) > latest else "missing",
            "target_date_utc": start.date().isoformat(),
            "source_latest_timestamp_utc": latest.isoformat(),
            "wind_curtailment_mwh": None,
            "solar_curtailment_mwh": None,
            "total_curtailment_mwh": None,
            "wind_share_percent": None,
            "solar_share_percent": None,
            "complete_half_hour_count": int(rows["source_labels_complete_flag"].sum()),
            "summary": "",
        }
        if start < self._data.index.min():
            result["status"] = "missing"
        if start in self._days.index:
            row = self._days.loc[start]
            wind, solar = _mwh(row["wind_curtailment_mwh"]), _mwh(row["solar_curtailment_mwh"])
            total = _mwh(wind + solar)
            result.update({
                "status": "available",
                "wind_curtailment_mwh": wind,
                "solar_curtailment_mwh": solar,
                "total_curtailment_mwh": total,
                "wind_share_percent": _percent(wind, total),
                "solar_share_percent": _percent(solar, total),
            })
        elif (
            len(rows) == HALF_HOURS_PER_DAY
            and rows["wind_reported_flag"].eq(1).all()
            and rows["solar_reported_flag"].eq(0).all()
        ):
            result.update({
                "status": "solar_not_published",
                "wind_curtailment_mwh": _mwh(rows["wind_curtailment_mwh"].sum()),
            })
        result["summary"] = _summary(result)
        if include_half_hours:
            result["half_hours"] = [
                {
                    "timestamp_utc": timestamp.isoformat(),
                    "wind_curtailment_mwh": None if pd.isna(wind) else _mwh(wind),
                    "solar_curtailment_mwh": None if pd.isna(solar) else _mwh(solar),
                }
                for timestamp, wind, solar in zip(rows.index, rows["wind_curtailment_mwh"], rows["solar_curtailment_mwh"])
            ]
        return result


def _summary(result: dict[str, Any]) -> str:
    day = result["target_date_utc"]
    if result["status"] == "available":
        if not result["total_curtailment_mwh"]:
            return f"No wind or solar power was curtailed in Ireland on {day} (UTC)."
        return (
            f"On {day} (UTC), {result['total_curtailment_mwh']:,.1f} MWh of renewable power was curtailed in Ireland: "
            f"{result['wind_share_percent']:g}% wind ({result['wind_curtailment_mwh']:,.1f} MWh) and "
            f"{result['solar_share_percent']:g}% solar ({result['solar_curtailment_mwh']:,.1f} MWh)."
        )
    if result["status"] == "solar_not_published":
        return (
            f"On {day} (UTC), {result['wind_curtailment_mwh']:,.1f} MWh of wind power was curtailed. EirGrid did not "
            "publish solar figures for Ireland before April 2023, so solar is unknown, not zero."
        )
    if result["status"] == "pending":
        return f"EirGrid has not yet published a complete day of figures for {day}. Try again after the archive is refreshed."
    return (
        f"EirGrid's wind and solar records for {day} are incomplete "
        f"({result['complete_half_hour_count']} of 48 half-hours), so no daily total is given."
    )
