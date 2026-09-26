"""Read-only EirGrid actual-value lookups keyed by a prediction's target time.

The bundled observation archive is a snapshot, not a live feed. Unknown future
targets remain pending until a refreshed archive is deployed; missing internal
rows are never silently interpreted as zero.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .daily_curtailment import DEFAULT_HISTORY, build_daily_labels


ACTUAL_COLUMNS = (
    "dispatch_down_mwh",
    "curtailment_mwh",
    "constraint_mwh",
)


class ActualsService:
    def __init__(self, dataset_path: Path | str = DEFAULT_HISTORY) -> None:
        self.dataset_path = Path(dataset_path)
        self._data: pd.DataFrame | None = None
        self._days: pd.DataFrame | None = None

    def _load(self) -> None:
        if self._data is not None:
            return
        columns = ["timestamp_utc", "dispatch_down_available_flag", *ACTUAL_COLUMNS]
        data = pd.read_csv(self.dataset_path, usecols=columns)
        data["timestamp_utc"] = pd.to_datetime(data["timestamp_utc"], utc=True, errors="raise")
        if data["timestamp_utc"].duplicated().any():
            raise ValueError("Actuals archive has duplicate UTC half-hours")
        for column in ACTUAL_COLUMNS:
            data[column] = pd.to_numeric(data[column], errors="coerce")
        data = data.sort_values("timestamp_utc").set_index("timestamp_utc", drop=False)
        if data.empty:
            raise ValueError("Actuals archive is empty")
        days = build_daily_labels(data.reset_index(drop=True))
        self._days = (
            days.set_index("issue_timestamp_utc")
            if not days.empty
            else pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))
        )
        self._data = data

    def coverage(self) -> dict[str, Any]:
        self._load()
        assert self._data is not None and self._days is not None
        return {
            "source": "EirGrid processed half-hour observation archive",
            "available_target_timestamp_min_utc": self._data.index.min().isoformat(),
            "available_target_timestamp_max_utc": self._data.index.max().isoformat(),
            "complete_day_min_utc": self._days.index.min().date().isoformat() if not self._days.empty else None,
            "complete_day_max_utc": self._days.index.max().date().isoformat() if not self._days.empty else None,
            "notice": "Snapshot only: future actuals require a refreshed EirGrid archive and redeployment.",
        }

    def point(self, target_timestamp: pd.Timestamp) -> dict[str, Any]:
        self._load()
        assert self._data is not None
        target = pd.Timestamp(target_timestamp)
        if target.tzinfo is None or target.utcoffset() is None:
            raise ValueError("A timezone is required for target_timestamp_utc")
        target = target.tz_convert("UTC")
        if target.minute not in (0, 30) or target.second or target.microsecond or target.nanosecond:
            raise ValueError("Target must align with a UTC half-hour boundary")
        latest = self._data.index.max()
        result: dict[str, Any] = {
            "status": "missing" if target <= latest else "pending",
            "target_timestamp_utc": target.isoformat(),
            "source_latest_timestamp_utc": latest.isoformat(),
            "actual_dispatch_down_mwh": None,
            "actual_curtailment_mwh": None,
            "actual_constraint_mwh": None,
            "actual_dispatch_down_event": None,
        }
        if target not in self._data.index:
            return result
        row = self._data.loc[target]
        values = row[list(ACTUAL_COLUMNS)].to_numpy(dtype=float)
        if row["dispatch_down_available_flag"] != 1 or not np.isfinite(values).all() or (values < 0).any():
            return result
        result.update({
            "status": "available",
            "actual_dispatch_down_mwh": float(row["dispatch_down_mwh"]),
            "actual_curtailment_mwh": float(row["curtailment_mwh"]),
            "actual_constraint_mwh": float(row["constraint_mwh"]),
            "actual_dispatch_down_event": bool(row["dispatch_down_mwh"] > 0),
        })
        return result

    def points(self, target_timestamps: Iterable[pd.Timestamp]) -> list[dict[str, Any]]:
        return [self.point(timestamp) for timestamp in target_timestamps]

    def day(self, target_date: date) -> dict[str, Any]:
        self._load()
        assert self._data is not None and self._days is not None
        target = pd.Timestamp(target_date, tz="UTC")
        latest = self._data.index.max()
        result: dict[str, Any] = {
            "status": "pending" if target.date() >= latest.date() else "missing",
            "target_date_utc": target.date().isoformat(),
            "source_latest_timestamp_utc": latest.isoformat(),
            "actual_curtailment_mwh": None,
            "actual_curtailment_event": None,
            "complete_half_hour_count": 0,
        }
        if target not in self._days.index:
            return result
        row = self._days.loc[target]
        result.update({
            "status": "available",
            "actual_curtailment_mwh": float(row["curtailment_mwh"]),
            "actual_curtailment_event": bool(row["curtailment_event"]),
            "complete_half_hour_count": 48,
        })
        return result
