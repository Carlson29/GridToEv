"""Technology-level (Wind/Solar) curtailment labels from EirGrid dispatch-down.

``build_core_history`` sums IE rows across ``UT_TYPE`` and so keeps only the
total. This module keeps a parallel, audited label table keyed by UTC
half-hour and technology. It never changes the core history or any model.

Missing-versus-zero rule: a technology with no reported row for a half-hour
is *unknown* (NaN), not zero. A reported numeric zero stays zero. Before
2023-04-01 EirGrid publishes no IE Solar rows, so those intervals are
wind-only and are not complete source labels.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .eirgrid_history import (
    ACCOUNTING_TOLERANCE_MWH,
    DataQualityError,
    normalise_dispatch_frame,
)


LABEL_JURISDICTION = "IE"
# Audited IE taxonomy for 2021-2026 workbooks. Any other category is rejected
# rather than folded into Solar or silently dropped.
SOURCE_TECHNOLOGIES = {"Wind": "wind", "Solar": "solar"}
LABEL_MEASURES = {
    "curtailment_mwh": "curtailment_mwh",
    "constraint_mwh": "constraint_mwh",
    "dispatch_down_mwh": "dispatch_down_mwh",
}
HALF_HOURS_PER_DAY = 48
RECONCILIATION_TOLERANCE_MWH = ACCOUNTING_TOLERANCE_MWH


def normalise_source_dispatch_frame(raw: pd.DataFrame, *, source_id: str) -> pd.DataFrame:
    """Validate one raw ``DD HH`` sheet and return long IE technology rows.

    ``normalise_dispatch_frame`` sums duplicate keys, which is acceptable for
    totals but would hide a corrupt source label. Duplicates are therefore
    rejected here before that shared GMT-offset normalisation runs.
    """

    if "JURISDICTION" not in raw or "UT_TYPE" not in raw:
        raise DataQualityError(f"{source_id} missing UT_TYPE/JURISDICTION columns")
    jurisdiction = raw["JURISDICTION"].astype("string").str.strip().str.upper()
    technology = raw["UT_TYPE"].astype("string").str.strip()
    ie_rows = jurisdiction.eq(LABEL_JURISDICTION)
    unknown = sorted(set(technology[ie_rows].dropna()) - set(SOURCE_TECHNOLOGIES))
    if unknown or technology[ie_rows].isna().any():
        raise DataQualityError(f"{source_id} has unaudited IE technology types: {unknown or ['<null>']}")
    raw_key = pd.DataFrame({
        "local": raw["HH_TIMESTAMP"], "offset": raw["GMT_OFFSET"],
        "jurisdiction": jurisdiction, "technology": technology,
    })
    duplicates = int(raw_key.duplicated().sum())
    if duplicates:
        raise DataQualityError(f"{source_id} has {duplicates} duplicate raw source-label keys")

    cleaned = raw.assign(JURISDICTION=jurisdiction, UT_TYPE=technology)
    frame = normalise_dispatch_frame(cleaned, source_id=source_id)
    frame = frame.loc[frame["jurisdiction"].eq(LABEL_JURISDICTION)]
    result = frame[["timestamp_utc", "jurisdiction", "technology_type", "source_id", *LABEL_MEASURES]].copy()
    values = result[list(LABEL_MEASURES)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise DataQualityError(f"{source_id} has missing or non-finite source labels")
    if (values < 0).any():
        raise DataQualityError(f"{source_id} has negative source labels")
    return result.reset_index(drop=True)


def build_source_label_table(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Pivot long technology rows into one row per contiguous UTC half-hour."""

    long = pd.concat(list(frames), ignore_index=True)
    if long.empty:
        raise DataQualityError("At least one source-label frame is required")
    key = ["timestamp_utc", "technology_type"]
    if long.duplicated(key).any():
        count = int(long.duplicated(key).sum())
        raise DataQualityError(f"Combined source labels have {count} duplicate UTC/technology keys")
    index = pd.date_range(
        long["timestamp_utc"].min(), long["timestamp_utc"].max(), freq="30min", tz="UTC", name="timestamp_utc",
    )
    table = pd.DataFrame(index=index)
    for technology, prefix in SOURCE_TECHNOLOGIES.items():
        rows = long.loc[long["technology_type"].eq(technology)].set_index("timestamp_utc")
        if not rows.index.isin(index).all():
            raise DataQualityError(f"{technology} labels are not on the 30-minute UTC grid")
        for measure, column in LABEL_MEASURES.items():
            table[f"{prefix}_{column}"] = rows[measure].reindex(index)
        table[f"{prefix}_reported_flag"] = table[f"{prefix}_curtailment_mwh"].notna().astype("int8")
    complete = table[[f"{prefix}_reported_flag" for prefix in SOURCE_TECHNOLOGIES.values()]].all(axis=1)
    table["source_labels_complete_flag"] = complete.astype("int8")
    # Only a complete taxonomy yields a total; wind-only intervals stay unknown.
    table["source_curtailment_total_mwh"] = (
        table["wind_curtailment_mwh"] + table["solar_curtailment_mwh"]
    ).where(complete)
    return table.reset_index()


def reconcile_with_core(labels: pd.DataFrame, core: pd.DataFrame) -> dict[str, object]:
    """Check complete Wind + Solar against the core history's IE curtailment total."""

    core_total = core.assign(timestamp_utc=pd.to_datetime(core["timestamp_utc"], utc=True))
    merged = labels.merge(
        core_total[["timestamp_utc", "curtailment_mwh"]].rename(columns={"curtailment_mwh": "core_curtailment_mwh"}),
        on="timestamp_utc", how="left", validate="one_to_one",
    )
    complete = merged["source_labels_complete_flag"].eq(1)
    comparable = complete & merged["core_curtailment_mwh"].notna()
    difference = (merged.loc[comparable, "source_curtailment_total_mwh"] - merged.loc[comparable, "core_curtailment_mwh"]).abs()
    # Wind-only intervals must still equal the core total because Solar was
    # not published; this verifies the absence is structural, not a lost row.
    wind_only = merged["wind_reported_flag"].eq(1) & merged["solar_reported_flag"].eq(0) & merged["core_curtailment_mwh"].notna()
    wind_only_difference = (merged.loc[wind_only, "wind_curtailment_mwh"] - merged.loc[wind_only, "core_curtailment_mwh"]).abs()
    report = {
        "tolerance_mwh": RECONCILIATION_TOLERANCE_MWH,
        "complete_rows_compared": int(comparable.sum()),
        "complete_rows_missing_core_total": int((complete & merged["core_curtailment_mwh"].isna()).sum()),
        "max_abs_difference_mwh": float(difference.max()) if not difference.empty else 0.0,
        "wind_only_rows_compared": int(wind_only.sum()),
        "wind_only_max_abs_difference_mwh": float(wind_only_difference.max()) if not wind_only_difference.empty else 0.0,
    }
    if report["complete_rows_missing_core_total"]:
        raise DataQualityError("Complete source labels exist where the core total is missing")
    if max(report["max_abs_difference_mwh"], report["wind_only_max_abs_difference_mwh"]) >= RECONCILIATION_TOLERANCE_MWH:
        raise DataQualityError(f"Source labels do not reconcile with core curtailment: {report}")
    return report


def build_daily_source_labels(labels: pd.DataFrame) -> pd.DataFrame:
    """Sum 48 consecutive complete UTC half-hours per day, like the V2 label."""

    data = labels.copy()
    data["timestamp_utc"] = pd.to_datetime(data["timestamp_utc"], utc=True)
    data["target_date_utc"] = data["timestamp_utc"].dt.floor("D")
    results = []
    for day, group in data.groupby("target_date_utc", sort=True):
        expected = pd.date_range(day, periods=HALF_HOURS_PER_DAY, freq="30min", tz="UTC")
        if len(group) != HALF_HOURS_PER_DAY or not group["timestamp_utc"].reset_index(drop=True).equals(pd.Series(expected)):
            continue
        if not group["source_labels_complete_flag"].eq(1).all():
            continue
        wind = float(group["wind_curtailment_mwh"].sum())
        solar = float(group["solar_curtailment_mwh"].sum())
        total = wind + solar
        results.append({
            "target_date_utc": day,
            "wind_curtailment_mwh": wind,
            "solar_curtailment_mwh": solar,
            "source_curtailment_total_mwh": total,
            "wind_share": wind / total if total > 0 else np.nan,
        })
    columns = ["target_date_utc", "wind_curtailment_mwh", "solar_curtailment_mwh", "source_curtailment_total_mwh", "wind_share"]
    return pd.DataFrame(results, columns=columns)


def reconcile_daily_with_v2(daily: pd.DataFrame, v2_dataset: pd.DataFrame) -> dict[str, object]:
    """Complete source days must sum to the existing V2 daily curtailment label."""

    v2 = v2_dataset[["issue_timestamp_utc", "curtailment_mwh"]].copy()
    v2["target_date_utc"] = pd.to_datetime(v2["issue_timestamp_utc"], utc=True)
    merged = v2.merge(daily, on="target_date_utc", how="left", validate="one_to_one")
    matched = merged["source_curtailment_total_mwh"].notna()
    difference = (merged.loc[matched, "source_curtailment_total_mwh"] - merged.loc[matched, "curtailment_mwh"]).abs()
    report = {
        "v2_days": int(len(merged)),
        "matched_days": int(matched.sum()),
        "v2_days_without_complete_source_labels": [
            day.date().isoformat() for day in merged.loc[~matched, "target_date_utc"]
        ],
        "max_abs_difference_mwh": float(difference.max()) if not difference.empty else 0.0,
    }
    if report["max_abs_difference_mwh"] >= RECONCILIATION_TOLERANCE_MWH:
        raise DataQualityError(f"Daily source labels do not reconcile with V2 labels: {report}")
    return report


def source_label_quality_report(labels: pd.DataFrame, daily: pd.DataFrame) -> dict[str, object]:
    """Per-year coverage, first/last valid timestamps and source shares."""

    data = labels.copy()
    data["year"] = data["timestamp_utc"].dt.year
    by_year = {}
    for year, group in data.groupby("year", sort=True):
        entry: dict[str, object] = {"half_hours": int(len(group))}
        for prefix in SOURCE_TECHNOLOGIES.values():
            reported = group.loc[group[f"{prefix}_reported_flag"].eq(1)]
            entry[prefix] = {
                "reported_half_hours": int(len(reported)),
                "missing_half_hours": int(len(group) - len(reported)),
                "first_reported_utc": reported["timestamp_utc"].min().isoformat() if len(reported) else None,
                "last_reported_utc": reported["timestamp_utc"].max().isoformat() if len(reported) else None,
                "curtailment_mwh": round(float(reported[f"{prefix}_curtailment_mwh"].sum()), 3),
                "positive_curtailment_half_hours": int(reported[f"{prefix}_curtailment_mwh"].gt(0).sum()),
            }
        complete = group.loc[group["source_labels_complete_flag"].eq(1)]
        total = float(complete["source_curtailment_total_mwh"].sum())
        entry["complete_half_hours"] = int(len(complete))
        entry["solar_share_of_complete_curtailment"] = (
            round(float(complete["solar_curtailment_mwh"].sum()) / total, 6) if total > 0 else None
        )
        by_year[str(year)] = entry
    complete = labels.loc[labels["source_labels_complete_flag"].eq(1), "timestamp_utc"]
    return {
        "jurisdiction": LABEL_JURISDICTION,
        "technologies": list(SOURCE_TECHNOLOGIES),
        "unit": "MWh per UTC half-hour",
        "missing_versus_zero_rule": (
            "An unreported technology interval is NaN (unknown), never zero. "
            "A reported numeric zero is kept as zero."
        ),
        "rows": int(len(labels)),
        "first_interval_utc": labels["timestamp_utc"].min().isoformat(),
        "last_interval_utc": labels["timestamp_utc"].max().isoformat(),
        "first_complete_interval_utc": complete.min().isoformat() if len(complete) else None,
        "last_complete_interval_utc": complete.max().isoformat() if len(complete) else None,
        "complete_half_hours": int(len(complete)),
        "duplicate_natural_keys": int(labels.duplicated("timestamp_utc").sum()),
        "by_year": by_year,
        "daily": {
            "complete_days": int(len(daily)),
            "first_day_utc": daily["target_date_utc"].min().date().isoformat() if len(daily) else None,
            "last_day_utc": daily["target_date_utc"].max().date().isoformat() if len(daily) else None,
            "positive_curtailment_days": int(daily["source_curtailment_total_mwh"].gt(0).sum()),
            "positive_solar_curtailment_days": int(daily["solar_curtailment_mwh"].gt(0).sum()),
        },
    }
