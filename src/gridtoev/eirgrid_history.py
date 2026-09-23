"""Normalise EirGrid system and renewable dispatch-down history."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


ACCOUNTING_TOLERANCE_MWH = 0.05


class DataQualityError(ValueError):
    """Raised when source data violates a modelling-safety invariant."""


SYSTEM_RENAME = {
    "NI Generation": "eirgrid_ni_generation_mw",
    "NI Demand": "eirgrid_ni_demand_mw",
    "NI Wind Availability": "eirgrid_ni_wind_availability_mw",
    "NI Wind Generation": "eirgrid_ni_wind_generation_mw",
    "NI Solar Availability": "eirgrid_ni_solar_availability_mw",
    "NI Solar Generation": "eirgrid_ni_solar_generation_mw",
    "Moyle I/C": "eirgrid_moyle_flow_mw",
    "NI Wind Penetration": "eirgrid_ni_wind_penetration_ratio",
    "NI Solar Penetration": "eirgrid_ni_solar_penetration_ratio",
    "IE Generation": "eirgrid_ie_generation_mw",
    "IE Demand": "eirgrid_ie_demand_mw",
    "IE Wind Availability": "eirgrid_ie_wind_availability_mw",
    "IE Wind Generation": "eirgrid_ie_wind_generation_mw",
    "IE Solar Availability": "eirgrid_ie_solar_availability_mw",
    "IE Solar Generation": "eirgrid_ie_solar_generation_mw",
    "IE Hydro": "eirgrid_ie_hydro_generation_mw",
    "EWIC I/C": "eirgrid_ewic_flow_mw",
    "Greenlink I/C": "eirgrid_greenlink_flow_mw",
    "IE Wind Penetration": "eirgrid_ie_wind_penetration_ratio",
    "IE Solar Penetration": "eirgrid_ie_solar_penetration_ratio",
    "AI Generation": "eirgrid_all_island_generation_mw",
    "AI Demand": "eirgrid_all_island_demand_mw",
    "AI Wind Availability": "eirgrid_all_island_wind_availability_mw",
    "AI Wind Generation": "eirgrid_all_island_wind_generation_mw",
    "AI Solar Availability": "eirgrid_all_island_solar_availability_mw",
    "AI Solar Generation": "eirgrid_all_island_solar_generation_mw",
    "AI Hydro": "eirgrid_all_island_hydro_generation_mw",
    "Inter-Jurisdictional Flow": "eirgrid_interjurisdictional_flow_mw",
    "AI Wind Penetration": "eirgrid_all_island_wind_penetration_ratio",
    "AI Solar Penetration": "eirgrid_all_island_solar_penetration_ratio",
    "AI Oversupply": "eirgrid_all_island_oversupply_mw",
    "AI Oversupply Percentage": "eirgrid_all_island_oversupply_ratio",
    "SNSP": "eirgrid_snsp_ratio",
}


DISPATCH_RENAME = {
    "Sum of AV_MWH": "available_renewable_mwh",
    "Sum of AO_MWH": "actual_renewable_output_mwh",
    "Sum of HI_FRQ_MIN_GEN_MWH": "high_frequency_min_generation_mwh",
    "Sum of ROCOF_INERTIA_MWH": "rocof_inertia_mwh",
    "Sum of SNSP_MWH": "snsp_curtailment_mwh",
    "Sum of TRANS_CONSTR_MWH": "transmission_constraint_mwh",
    "Sum of TSO_TEST_MWH": "tso_test_mwh",
    "Sum of DD_MWH": "dispatch_down_mwh",
    "Sum of CURTAILMENTS_MWH": "curtailment_mwh",
    "Sum of CONSTRAINTS_MWH": "constraint_mwh",
    "Sum of OTHER_MWH": "other_reduction_mwh",
}


def _normalise_utc(
    local_values: pd.Series,
    offset_values: pd.Series,
    *,
    source_id: str,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    local = pd.to_datetime(local_values, errors="coerce")
    offset = pd.to_numeric(offset_values, errors="coerce")
    invalid = local.isna() | offset.isna()
    if invalid.any():
        raise DataQualityError(
            f"{source_id} has {int(invalid.sum())} invalid local timestamps/GMT offsets"
        )
    utc = (local - pd.to_timedelta(offset, unit="h")).dt.tz_localize("UTC")
    original = local.dt.strftime("%Y-%m-%dT%H:%M:%S")
    return utc, original, offset


def normalise_system_frame(raw: pd.DataFrame, *, source_id: str) -> pd.DataFrame:
    required = {"DateTime", "GMT Offset"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise DataQualityError(f"{source_id} missing system columns: {missing}")

    available_values = [column for column in SYSTEM_RENAME if column in raw.columns]
    if not available_values:
        raise DataQualityError(f"{source_id} has no recognised system measurements")
    timestamp, original, offset = _normalise_utc(
        raw["DateTime"], raw["GMT Offset"], source_id=source_id
    )
    result = pd.DataFrame(
        {
            "timestamp_utc": timestamp,
            "timestamp_local_original": original,
            "gmt_offset_hours": offset,
            "source_id": source_id,
        }
    )
    for source_column in available_values:
        result[SYSTEM_RENAME[source_column]] = pd.to_numeric(
            raw[source_column], errors="coerce"
        )
    result = result.sort_values("timestamp_utc", kind="stable").reset_index(drop=True)
    duplicates = int(result.duplicated("timestamp_utc").sum())
    if duplicates:
        raise DataQualityError(
            f"{source_id} contains {duplicates} duplicate UTC timestamps"
        )
    return result


def read_system_workbook(path: Path, *, source_id: str) -> pd.DataFrame:
    return normalise_system_frame(
        pd.read_excel(path, sheet_name="System Data"), source_id=source_id
    )


def normalise_dispatch_frame(raw: pd.DataFrame, *, source_id: str) -> pd.DataFrame:
    required = {
        "UT_TYPE",
        "JURISDICTION",
        "HH_TIMESTAMP",
        "GMT_OFFSET",
        "Sum of DD_MWH",
        "Sum of CURTAILMENTS_MWH",
        "Sum of CONSTRAINTS_MWH",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise DataQualityError(f"{source_id} missing dispatch columns: {missing}")

    timestamp, original, offset = _normalise_utc(
        raw["HH_TIMESTAMP"], raw["GMT_OFFSET"], source_id=source_id
    )
    result = pd.DataFrame(
        {
            "timestamp_utc": timestamp,
            "timestamp_local_original": original,
            "gmt_offset_hours": offset,
            "jurisdiction": raw["JURISDICTION"].astype("string").str.upper(),
            "technology_type": raw["UT_TYPE"].astype("string"),
            "source_id": source_id,
        }
    )
    for source_column, output_column in DISPATCH_RENAME.items():
        if source_column in raw.columns:
            result[output_column] = pd.to_numeric(raw[source_column], errors="coerce")
        else:
            result[output_column] = float("nan")

    key = ["timestamp_utc", "jurisdiction", "technology_type"]
    numeric_columns = list(DISPATCH_RENAME.values())
    if result.duplicated(key).any():
        result = (
            result.groupby(key, as_index=False, dropna=False)
            .agg(
                {
                    "timestamp_local_original": "first",
                    "gmt_offset_hours": "first",
                    "source_id": "first",
                    **{column: "sum" for column in numeric_columns},
                }
            )
        )
    result = result.sort_values(key, kind="stable").reset_index(drop=True)

    comparable = result[
        ["dispatch_down_mwh", "curtailment_mwh", "constraint_mwh"]
    ].notna().all(axis=1)
    difference = (
        result.loc[comparable, "dispatch_down_mwh"]
        - result.loc[comparable, "curtailment_mwh"]
        - result.loc[comparable, "constraint_mwh"]
    ).abs()
    if not difference.empty and float(difference.max()) >= ACCOUNTING_TOLERANCE_MWH:
        raise DataQualityError(
            f"{source_id} dispatch-down accounting differs by up to "
            f"{float(difference.max()):.6f} MWh"
        )
    return result


def read_dispatch_workbook(path: Path, *, source_id: str) -> pd.DataFrame:
    return normalise_dispatch_frame(
        pd.read_excel(path, sheet_name="DD HH"), source_id=source_id
    )


def _frame_coverage(frame: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for source_id, source in frame.groupby("source_id", sort=True):
        records.append(
            {
                "source_id": source_id,
                "rows": int(len(source)),
                "first_interval_utc": source["timestamp_utc"].min().isoformat(),
                "last_interval_utc": source["timestamp_utc"].max().isoformat(),
                "available_columns": sorted(
                    column
                    for column in source.columns
                    if column
                    not in {
                        "timestamp_utc",
                        "timestamp_local_original",
                        "gmt_offset_hours",
                        "source_id",
                    }
                    and source[column].notna().any()
                ),
            }
        )
    return records


def build_core_history(
    system_frames: Iterable[pd.DataFrame],
    dispatch_frames: Iterable[pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Create a half-hourly core table without dropping optional-source gaps."""

    system_list = list(system_frames)
    dispatch_list = list(dispatch_frames)
    if not system_list:
        raise DataQualityError("At least one system-history frame is required")
    system = pd.concat(system_list, ignore_index=True, sort=False)
    if system.duplicated("timestamp_utc").any():
        count = int(system.duplicated("timestamp_utc").sum())
        raise DataQualityError(f"Combined system history has {count} duplicate UTC keys")

    system_value_columns = sorted(
        column for column in system.columns if column.startswith("eirgrid_")
    )
    system_half_hour = (
        system.set_index("timestamp_utc")[system_value_columns]
        .sort_index()
        .resample("30min")
        .mean()
    )
    system_observed = (
        system.set_index("timestamp_utc")
        .assign(_observed=1)["_observed"]
        .resample("30min")
        .max()
    )
    system_half_hour["eirgrid_system_available_flag"] = (
        system_observed.fillna(0).astype("int8")
    )

    dispatch_coverage: list[dict[str, object]] = []
    if dispatch_list:
        dispatch = pd.concat(dispatch_list, ignore_index=True, sort=False)
        dispatch_coverage = _frame_coverage(dispatch)
        dispatch_ie = dispatch.loc[dispatch["jurisdiction"].eq("IE")].copy()
        dispatch_value_columns = list(DISPATCH_RENAME.values())
        dispatch_half_hour = (
            dispatch_ie.groupby("timestamp_utc", as_index=True)[dispatch_value_columns]
            .sum(min_count=1)
            .sort_index()
        )
        dispatch_half_hour["dispatch_down_available_flag"] = 1
    else:
        dispatch_half_hour = pd.DataFrame(
            columns=[*DISPATCH_RENAME.values(), "dispatch_down_available_flag"],
            index=pd.DatetimeIndex([], tz="UTC", name="timestamp_utc"),
        )

    first_timestamp = min(system_half_hour.index.min(), dispatch_half_hour.index.min()) \
        if not dispatch_half_hour.empty else system_half_hour.index.min()
    last_timestamp = max(system_half_hour.index.max(), dispatch_half_hour.index.max()) \
        if not dispatch_half_hour.empty else system_half_hour.index.max()
    index = pd.date_range(first_timestamp, last_timestamp, freq="30min", tz="UTC")
    index.name = "timestamp_utc"
    core = system_half_hour.reindex(index).join(dispatch_half_hour.reindex(index))
    core["eirgrid_system_available_flag"] = (
        core["eirgrid_system_available_flag"].fillna(0).astype("int8")
    )
    core["dispatch_down_available_flag"] = (
        core["dispatch_down_available_flag"].fillna(0).astype("int8")
    )
    # Schema availability changed materially over 2021-2026 (for example,
    # Ireland solar and Greenlink arrived later). Per-feature flags let models
    # distinguish "not published then" from an ordinary numeric value.
    for column in system_value_columns:
        core[f"{column}_available_flag"] = core[column].notna().astype("int8")
    core = core.reset_index()

    accounting_rows = core[
        ["dispatch_down_mwh", "curtailment_mwh", "constraint_mwh"]
    ].notna().all(axis=1)
    accounting_difference = (
        core.loc[accounting_rows, "dispatch_down_mwh"]
        - core.loc[accounting_rows, "curtailment_mwh"]
        - core.loc[accounting_rows, "constraint_mwh"]
    ).abs()
    max_difference = (
        float(accounting_difference.max()) if not accounting_difference.empty else 0.0
    )
    if max_difference >= ACCOUNTING_TOLERANCE_MWH:
        raise DataQualityError(
            f"Core dispatch-down accounting differs by up to {max_difference:.6f} MWh"
        )

    span_days = (core["timestamp_utc"].max() - core["timestamp_utc"].min()).days
    quality: dict[str, object] = {
        "rows": int(len(core)),
        "columns": int(core.shape[1]),
        "first_interval_utc": core["timestamp_utc"].min().isoformat(),
        "last_interval_utc": core["timestamp_utc"].max().isoformat(),
        "coverage_days": span_days,
        "meets_24_month_target": span_days >= 730,
        "duplicate_natural_keys": int(core.duplicated("timestamp_utc").sum()),
        "max_dispatch_accounting_difference_mwh": max_difference,
        "system_available_rows": int(core["eirgrid_system_available_flag"].sum()),
        "dispatch_down_available_rows": int(core["dispatch_down_available_flag"].sum()),
        "missing_cells_by_column": {
            column: int(count)
            for column, count in core.isna().sum().items()
            if int(count) > 0
        },
        "system_source_coverage": _frame_coverage(system),
        "dispatch_source_coverage": dispatch_coverage,
        "feature_columns": system_value_columns,
    }
    return core, quality
