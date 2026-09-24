"""Leakage-safe EirGrid outage and ECP constraint features.

The public workbooks are publication snapshots.  Feature construction therefore
selects the newest snapshot that was already public at ``issue_timestamp_utc``
before evaluating whether an outage is active at the prediction target.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

from gridtoev.constants import IDENTIFIER_COLUMNS, TARGET_COLUMNS
from gridtoev.eirgrid_history import DataQualityError


OUTAGE_SCHEMA_VERSION = "eirgrid-outages-v1"
ECP_SCHEMA_VERSION = "eirgrid-ecp-v1"

REGION_SHEETS = {
    "GEN_AreaSouthWest": "area_south_west",
    "GEN_SouthEast": "south_east",
    "GEN_NorthWest": "north_west",
    "GEN_NorthEast": "north_east",
    "GEN_Dublin": "dublin",
    "GEN_Cork": "cork",
}
TRANSMISSION_REGIONS = tuple(REGION_SHEETS.values())
ECP_AREAS = ("A", "B", "C", "D", "E", "F", "G", "H1", "H2", "I", "J", "K")

OUTAGE_COLUMNS = [
    "outage_id",
    "source_family",
    "snapshot_id",
    "published_at_utc",
    "start_timestamp_utc",
    "end_timestamp_utc",
    "outage_type",
    "asset_name",
    "unit_id",
    "region",
    "constraint_group",
    "rated_mw",
    "unavailable_mw",
    "voltage_kv",
    "source_url",
    "source_file",
    "schema_version",
]

ECP_COLUMNS = [
    "constraint_group",
    "node",
    "study_year",
    "scenario",
    "generation_type",
    "installed_mw",
    "available_energy_gwh",
    "constraint_gwh",
    "constraint_ratio",
    "snapshot_id",
    "published_at_utc",
    "source_url",
    "source_file",
    "schema_version",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _local_series_to_utc(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce", dayfirst=True)
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        return parsed.dt.tz_convert("UTC")
    return parsed.dt.tz_localize(
        "Europe/Dublin", ambiguous="NaT", nonexistent="shift_forward"
    ).dt.tz_convert("UTC")


def _snapshot_id(family: str, published: pd.Timestamp, path: Path) -> str:
    return f"{family}:{published.strftime('%Y%m%dT%H%M%SZ')}:{_sha256(path)[:12]}"


def parse_generation_outage_plan(
    path: Path | str,
    *,
    published_at: object,
    source_url: str,
) -> pd.DataFrame:
    """Parse an All-Island Generation Outage Plan ``Outage Dates`` sheet.

    ``F`` means a full forced outage and ``S`` a full scheduled outage.  A
    numeric value is the unit capacity remaining during a derating, so
    unavailable MW is ``NPR - Value``.
    """

    source_path = Path(path)
    raw = pd.read_excel(
        source_path,
        sheet_name="Outage Dates",
        header=3,
        usecols="B:H",
    )
    raw.columns = [str(column).strip() for column in raw.columns]
    required = {"Station", "Unit", "NPR", "Start Date", "End Date", "Value"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise DataQualityError(f"Generation outage workbook is missing columns: {missing}")

    result = raw.loc[raw["Unit"].notna()].copy()
    result["rated_mw"] = pd.to_numeric(result["NPR"], errors="coerce")
    result["start_timestamp_utc"] = _local_series_to_utc(result["Start Date"])
    result["end_timestamp_utc"] = _local_series_to_utc(result["End Date"])
    result = result.loc[
        result["rated_mw"].notna()
        & result["start_timestamp_utc"].notna()
        & result["end_timestamp_utc"].notna()
        & result["end_timestamp_utc"].gt(result["start_timestamp_utc"])
    ].copy()

    value_text = result["Value"].astype(str).str.strip().str.upper()
    numeric_value = pd.to_numeric(result["Value"], errors="coerce")
    result["outage_type"] = np.select(
        [value_text.eq("F"), value_text.eq("S"), numeric_value.notna()],
        ["forced", "planned", "derating"],
        default="unknown",
    )
    result["unavailable_mw"] = np.where(
        numeric_value.notna(),
        (result["rated_mw"] - numeric_value).clip(lower=0),
        result["rated_mw"],
    )
    published = _utc_timestamp(published_at)
    result["source_family"] = "generation_plan"
    result["snapshot_id"] = _snapshot_id("generation_plan", published, source_path)
    result["published_at_utc"] = published
    result["asset_name"] = result["Station"].astype(str).str.strip()
    result["unit_id"] = result["Unit"].astype(str).str.strip()
    result["region"] = np.where(
        result["unit_id"].str.upper().str.startswith("NI"),
        "northern_ireland",
        "ireland",
    )
    result["constraint_group"] = result["region"]
    result["voltage_kv"] = np.nan
    result["source_url"] = source_url
    result["source_file"] = source_path.name
    result["schema_version"] = OUTAGE_SCHEMA_VERSION
    occurrence = result.groupby("unit_id", sort=False).cumcount().astype(str)
    result["outage_id"] = "generation:" + result["unit_id"] + ":" + occurrence
    return result[OUTAGE_COLUMNS].reset_index(drop=True)


def _transmission_region_map(
    workbook: pd.ExcelFile,
) -> dict[str, str]:
    regions: dict[str, set[str]] = {}
    for sheet_name, region in REGION_SHEETS.items():
        if sheet_name not in workbook.sheet_names:
            continue
        frame = pd.read_excel(workbook, sheet_name=sheet_name, usecols="A:H")
        if "Outage ID" not in frame:
            continue
        for outage_id in frame["Outage ID"].dropna().astype(str).str.strip():
            regions.setdefault(outage_id, set()).add(region)
    return {
        outage_id: "|".join(sorted(values))
        for outage_id, values in regions.items()
    }


def parse_transmission_outage_programme(
    path: Path | str,
    *,
    published_at: object,
    source_url: str,
) -> pd.DataFrame:
    """Parse the all-region tab in an EirGrid Transmission Outage Programme.

    Published finish dates are treated as inclusive calendar days and converted
    to an exclusive midnight boundary by adding one day.
    """

    source_path = Path(path)
    with pd.ExcelFile(source_path) as workbook:
        if "GEN_ALL" not in workbook.sheet_names:
            raise DataQualityError("Transmission workbook is missing the GEN_ALL sheet")
        raw = pd.read_excel(workbook, sheet_name="GEN_ALL", usecols="A:H")
        region_map = _transmission_region_map(workbook)
    raw.columns = [str(column).strip() for column in raw.columns]
    required = {"Outage ID", "Feeder ID", "Outage Status", "Start", "Finish"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise DataQualityError(f"Transmission workbook is missing columns: {missing}")

    result = raw.loc[raw["Outage ID"].notna()].copy()
    result["start_timestamp_utc"] = _local_series_to_utc(result["Start"])
    result["end_timestamp_utc"] = _local_series_to_utc(result["Finish"]) + pd.Timedelta(days=1)
    result = result.loc[
        result["start_timestamp_utc"].notna()
        & result["end_timestamp_utc"].notna()
        & result["end_timestamp_utc"].gt(result["start_timestamp_utc"])
    ].copy()
    result["outage_id"] = result["Outage ID"].astype(str).str.strip()
    status = result["Outage Status"].astype(str).str.strip().str.lower()
    result["outage_type"] = status.map(
        {
            "scheduled": "planned",
            "proposed": "planned_proposed",
            "reference": "reference",
        }
    ).fillna(status.str.replace(r"\s+", "_", regex=True))
    result["asset_name"] = result["Feeder ID"].astype(str).str.strip()
    result["unit_id"] = result["outage_id"]
    result["region"] = result["outage_id"].map(region_map).fillna("unmapped")
    result["constraint_group"] = result["region"]
    result["voltage_kv"] = pd.to_numeric(
        result["asset_name"].str.extract(r"(?i)(\d{2,3})\s*kV", expand=False),
        errors="coerce",
    )
    result["rated_mw"] = np.nan
    result["unavailable_mw"] = np.nan
    published = _utc_timestamp(published_at)
    result["source_family"] = "transmission_programme"
    result["snapshot_id"] = _snapshot_id("transmission_programme", published, source_path)
    result["published_at_utc"] = published
    result["source_url"] = source_url
    result["source_file"] = source_path.name
    result["schema_version"] = OUTAGE_SCHEMA_VERSION
    result = result.drop_duplicates(["snapshot_id", "outage_id"], keep="last")
    return result[OUTAGE_COLUMNS].reset_index(drop=True)


def parse_ecp_constraint_report(
    path: Path | str,
    *,
    published_at: object,
    source_url: str,
) -> pd.DataFrame:
    """Parse nodal constraint results from the ECP publication workbook."""

    source_path = Path(path)
    raw = pd.read_excel(source_path, sheet_name="Dispatch Down Results", header=1)
    raw.columns = [str(column).strip() for column in raw.columns]
    rename = {
        "Area": "constraint_group",
        "Node": "node",
        "Year": "study_year",
        "Generation Scenario": "scenario",
        "Generation Type": "generation_type",
        "Installed (MW)": "installed_mw",
        "Available Energy (GWh)": "available_energy_gwh",
        "Constraint (GWh)": "constraint_gwh",
        "Constraint (%)": "constraint_ratio",
    }
    missing = sorted(set(rename) - set(raw.columns))
    if missing:
        raise DataQualityError(f"ECP constraint workbook is missing columns: {missing}")
    result = raw.rename(columns=rename)[list(rename.values())].copy()
    result["constraint_group"] = result["constraint_group"].astype(str).str.strip().str.upper()
    result = result.loc[result["constraint_group"].isin(ECP_AREAS)].copy()
    for column in (
        "study_year",
        "installed_mw",
        "available_energy_gwh",
        "constraint_gwh",
        "constraint_ratio",
    ):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.loc[
        result["installed_mw"].notna()
        & result["constraint_ratio"].notna()
        & result["node"].notna()
    ].copy()
    published = _utc_timestamp(published_at)
    result["snapshot_id"] = _snapshot_id("ecp_constraints", published, source_path)
    result["published_at_utc"] = published
    result["source_url"] = source_url
    result["source_file"] = source_path.name
    result["schema_version"] = ECP_SCHEMA_VERSION
    result["node"] = result["node"].astype(str).str.strip().str.lower()
    result["scenario"] = result["scenario"].astype(str).str.strip()
    result["generation_type"] = result["generation_type"].astype(str).str.strip().str.lower()
    return result[ECP_COLUMNS].reset_index(drop=True)


def deduplicate_outage_snapshots(outages: pd.DataFrame) -> pd.DataFrame:
    """Keep one asset event per publication snapshot."""

    if outages.empty:
        return pd.DataFrame(columns=OUTAGE_COLUMNS)
    result = outages.copy()
    for column in ("published_at_utc", "start_timestamp_utc", "end_timestamp_utc"):
        result[column] = pd.to_datetime(result[column], utc=True, errors="raise")
    result = result.sort_values(
        ["source_family", "published_at_utc", "snapshot_id", "outage_id"]
    ).drop_duplicates(["source_family", "snapshot_id", "outage_id"], keep="last")
    return result[OUTAGE_COLUMNS].reset_index(drop=True)


def _latest_snapshot(frame: pd.DataFrame, issue: pd.Timestamp) -> tuple[pd.DataFrame, pd.Timestamp | pd.NaT]:
    eligible = frame.loc[frame["published_at_utc"].le(issue)]
    if eligible.empty:
        return frame.iloc[0:0], pd.NaT
    published = eligible["published_at_utc"].max()
    return eligible.loc[eligible["published_at_utc"].eq(published)], published


def _hours(delta: pd.Timedelta | object) -> float:
    if pd.isna(delta):
        return np.nan
    return float(pd.Timedelta(delta).total_seconds() / 3600)


def _generation_features(
    snapshot: pd.DataFrame,
    issue: pd.Timestamp,
    target: pd.Timestamp,
) -> dict[str, float]:
    names = {
        "generation_planned_unavailable_mw": np.nan,
        "generation_forced_unavailable_mw": np.nan,
        "generation_derated_unavailable_mw": np.nan,
        "generation_total_unavailable_mw": np.nan,
        "generation_active_outage_count": np.nan,
        "generation_active_planned_count": np.nan,
        "generation_active_forced_count": np.nan,
        "generation_largest_unavailable_unit_mw": np.nan,
        "generation_hours_to_next_outage_start": np.nan,
        "generation_hours_to_first_active_outage_end": np.nan,
    }
    if snapshot.empty:
        return names
    active = snapshot.loc[
        snapshot["start_timestamp_utc"].le(target)
        & snapshot["end_timestamp_utc"].gt(target)
    ].copy()
    unavailable = pd.to_numeric(active["unavailable_mw"], errors="coerce").fillna(0.0)
    planned = active["outage_type"].isin(["planned", "planned_proposed"])
    forced = active["outage_type"].eq("forced")
    derating = active["outage_type"].eq("derating")
    future_starts = snapshot.loc[snapshot["start_timestamp_utc"].gt(issue), "start_timestamp_utc"]
    names.update(
        {
            "generation_planned_unavailable_mw": float(unavailable.loc[planned].sum()),
            "generation_forced_unavailable_mw": float(unavailable.loc[forced].sum()),
            "generation_derated_unavailable_mw": float(unavailable.loc[derating].sum()),
            "generation_total_unavailable_mw": float(unavailable.sum()),
            "generation_active_outage_count": int(len(active)),
            "generation_active_planned_count": int(planned.sum()),
            "generation_active_forced_count": int(forced.sum()),
            "generation_largest_unavailable_unit_mw": float(unavailable.max()) if len(active) else 0.0,
            "generation_hours_to_next_outage_start": (
                _hours(future_starts.min() - issue) if not future_starts.empty else np.nan
            ),
            "generation_hours_to_first_active_outage_end": (
                _hours(active["end_timestamp_utc"].min() - issue) if not active.empty else np.nan
            ),
        }
    )
    return names


def _transmission_features(
    snapshot: pd.DataFrame,
    issue: pd.Timestamp,
    target: pd.Timestamp,
) -> dict[str, float]:
    names = {
        "transmission_active_outage_count": np.nan,
        "transmission_active_scheduled_count": np.nan,
        "transmission_voltage_weighted_pressure": np.nan,
        "transmission_largest_voltage_kv": np.nan,
        "transmission_hours_to_next_outage_start": np.nan,
        "transmission_hours_to_first_active_outage_end": np.nan,
        **{
            f"transmission_active_outage_count_{region}": np.nan
            for region in TRANSMISSION_REGIONS
        },
    }
    if snapshot.empty:
        return names
    active = snapshot.loc[
        snapshot["start_timestamp_utc"].le(target)
        & snapshot["end_timestamp_utc"].gt(target)
    ].copy()
    voltage = pd.to_numeric(active["voltage_kv"], errors="coerce")
    future_starts = snapshot.loc[snapshot["start_timestamp_utc"].gt(issue), "start_timestamp_utc"]
    names.update(
        {
            "transmission_active_outage_count": int(len(active)),
            "transmission_active_scheduled_count": int(
                active["outage_type"].isin(["planned", "planned_proposed"]).sum()
            ),
            "transmission_voltage_weighted_pressure": float((voltage.fillna(0) / 400).sum()),
            "transmission_largest_voltage_kv": float(voltage.max()) if voltage.notna().any() else 0.0,
            "transmission_hours_to_next_outage_start": (
                _hours(future_starts.min() - issue) if not future_starts.empty else np.nan
            ),
            "transmission_hours_to_first_active_outage_end": (
                _hours(active["end_timestamp_utc"].min() - issue) if not active.empty else np.nan
            ),
        }
    )
    for region in TRANSMISSION_REGIONS:
        names[f"transmission_active_outage_count_{region}"] = int(
            active["region"].fillna("").str.split("|").map(lambda values: region in values).sum()
        )
    return names


def _weighted_ratio(frame: pd.DataFrame) -> float:
    if frame.empty:
        return np.nan
    weights = pd.to_numeric(frame["installed_mw"], errors="coerce").fillna(0.0)
    ratios = pd.to_numeric(frame["constraint_ratio"], errors="coerce")
    usable = ratios.notna() & weights.gt(0)
    if usable.any():
        return float(np.average(ratios.loc[usable], weights=weights.loc[usable]))
    return float(ratios.mean()) if ratios.notna().any() else np.nan


def _constraint_features(snapshot: pd.DataFrame) -> dict[str, float]:
    names = {
        "ecp_constraint_pressure_weighted_ratio": np.nan,
        "ecp_constraint_peak_group_ratio": np.nan,
        **{
            f"ecp_area_{area.lower()}_constraint_pressure_ratio": np.nan
            for area in ECP_AREAS
        },
        **{
            f"ecp_area_{area.lower()}_installed_mw": np.nan
            for area in ECP_AREAS
        },
    }
    if snapshot.empty:
        return names
    names["ecp_constraint_pressure_weighted_ratio"] = _weighted_ratio(snapshot)
    group_ratios = snapshot.groupby("constraint_group", sort=True).apply(
        _weighted_ratio, include_groups=False
    )
    names["ecp_constraint_peak_group_ratio"] = (
        float(group_ratios.max()) if not group_ratios.empty else np.nan
    )
    for area in ECP_AREAS:
        group = snapshot.loc[snapshot["constraint_group"].eq(area)]
        names[f"ecp_area_{area.lower()}_constraint_pressure_ratio"] = _weighted_ratio(group)
        names[f"ecp_area_{area.lower()}_installed_mw"] = (
            float(pd.to_numeric(group["installed_mw"], errors="coerce").sum())
            if not group.empty
            else np.nan
        )
    return names


def validate_outage_feature_matrix(table: pd.DataFrame) -> dict[str, int]:
    required = {
        "issue_timestamp_utc",
        "target_timestamp_utc",
        "forecast_horizon_minutes",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise DataQualityError(f"Outage feature table is missing columns: {missing}")
    frame = table.copy()
    issue = pd.to_datetime(frame["issue_timestamp_utc"], utc=True, errors="raise")
    target = pd.to_datetime(frame["target_timestamp_utc"], utc=True, errors="raise")
    expected = issue + pd.to_timedelta(frame["forecast_horizon_minutes"], unit="m")
    duplicates = int(
        frame.duplicated(["issue_timestamp_utc", "forecast_horizon_minutes"]).sum()
    )
    horizon_violations = int(target.ne(expected).sum())
    future_publications = 0
    for column in frame.columns:
        if column.endswith("_published_at_utc"):
            published = pd.to_datetime(frame[column], utc=True, errors="coerce")
            future_publications += int((published.notna() & published.gt(issue)).sum())
    numeric = frame.select_dtypes(include=["number"])
    nonfinite = int(np.isinf(numeric.to_numpy(dtype=float)).sum()) if not numeric.empty else 0
    checks = {
        "duplicate_natural_keys": duplicates,
        "horizon_alignment_violations": horizon_violations,
        "future_publication_violations": future_publications,
        "nonfinite_numeric_cells": nonfinite,
    }
    failed = {name: value for name, value in checks.items() if value}
    if failed:
        raise DataQualityError(f"Outage feature validation failed: {failed}")
    return checks


def _dictionary(table: pd.DataFrame) -> pd.DataFrame:
    identifiers = {
        "issue_timestamp_utc",
        "target_timestamp_utc",
        "forecast_horizon_minutes",
    }
    rows: list[dict[str, object]] = []
    for column in table.columns:
        if column in identifiers:
            source = "model row"
            role = "identifier"
            availability = "Always present in the modelling key."
        elif column.startswith("generation_"):
            source = "EirGrid All-Island Generation Outage Plan"
            role = "feature"
            availability = "Newest generation-plan snapshot published no later than issue time."
        elif column.startswith("transmission_"):
            source = "EirGrid Transmission Outage Programme"
            role = "feature"
            availability = "Newest transmission snapshot published no later than issue time."
        else:
            source = "EirGrid ECP constraint forecast workbook"
            role = "feature"
            availability = "Newest ECP snapshot published no later than issue time."
        if column.endswith("_mw"):
            unit = "MW"
        elif column.endswith("_kv"):
            unit = "kV"
        elif "hours_" in column or column.endswith("_age_hours"):
            unit = "hours"
        elif column.endswith("_ratio"):
            unit = "ratio"
        elif column.endswith("_count") or "_count_" in column or column.endswith("_flag"):
            unit = "count"
        elif column.endswith("_utc"):
            unit = "UTC timestamp"
        else:
            unit = "unitless"
        rows.append(
            {
                "column": column,
                "role": role,
                "dtype": str(table[column].dtype),
                "unit": unit,
                "source": source,
                "formula": "See docs/OUTAGE_CONSTRAINT_SIGNALS.md.",
                "availability_rule": availability,
                "missingness_handling": (
                    "Timestamp stays null; availability flag is zero."
                    if column.endswith("_utc")
                    else "Null when no eligible publication exists; availability flag distinguishes unknown from zero."
                ),
            }
        )
    return pd.DataFrame(rows)


def build_outage_constraint_features(
    modelling_rows: pd.DataFrame,
    outages: pd.DataFrame,
    constraints: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Build one causal outage/constraint feature row per issue and horizon."""

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

    outage_frame = deduplicate_outage_snapshots(outages)
    generation = outage_frame.loc[outage_frame["source_family"].eq("generation_plan")]
    transmission = outage_frame.loc[
        outage_frame["source_family"].eq("transmission_programme")
    ]
    constraint_frame = constraints.copy()
    if constraint_frame.empty:
        constraint_frame = pd.DataFrame(columns=ECP_COLUMNS)
    else:
        constraint_frame["published_at_utc"] = pd.to_datetime(
            constraint_frame["published_at_utc"], utc=True, errors="raise"
        )

    rows: list[dict[str, object]] = []
    for model_row in result.itertuples(index=False):
        issue = model_row.issue_timestamp_utc
        target = model_row.target_timestamp_utc
        generation_snapshot, generation_published = _latest_snapshot(generation, issue)
        transmission_snapshot, transmission_published = _latest_snapshot(transmission, issue)
        constraint_snapshot, constraint_published = _latest_snapshot(constraint_frame, issue)
        row: dict[str, object] = {
            "issue_timestamp_utc": issue,
            "target_timestamp_utc": target,
            "forecast_horizon_minutes": int(model_row.forecast_horizon_minutes),
            "generation_snapshot_published_at_utc": generation_published,
            "generation_source_available_flag": int(not generation_snapshot.empty),
            "generation_snapshot_age_hours": (
                _hours(issue - generation_published) if pd.notna(generation_published) else np.nan
            ),
            "transmission_snapshot_published_at_utc": transmission_published,
            "transmission_source_available_flag": int(not transmission_snapshot.empty),
            "transmission_snapshot_age_hours": (
                _hours(issue - transmission_published) if pd.notna(transmission_published) else np.nan
            ),
            "ecp_snapshot_published_at_utc": constraint_published,
            "ecp_source_available_flag": int(not constraint_snapshot.empty),
            "ecp_snapshot_age_hours": (
                _hours(issue - constraint_published) if pd.notna(constraint_published) else np.nan
            ),
        }
        row.update(_generation_features(generation_snapshot, issue, target))
        row.update(_transmission_features(transmission_snapshot, issue, target))
        row.update(_constraint_features(constraint_snapshot))
        row["generation_outage_missing_flag"] = int(generation_snapshot.empty)
        row["transmission_outage_missing_flag"] = int(transmission_snapshot.empty)
        row["ecp_constraint_missing_flag"] = int(constraint_snapshot.empty)
        rows.append(row)

    feature_table = pd.DataFrame(rows)
    checks = validate_outage_feature_matrix(feature_table)
    checks.update(
        {
            "rows": int(len(feature_table)),
            "columns": int(len(feature_table.columns)),
            "first_issue_utc": feature_table["issue_timestamp_utc"].min().isoformat(),
            "last_issue_utc": feature_table["issue_timestamp_utc"].max().isoformat(),
            "generation_source_coverage": float(
                feature_table["generation_source_available_flag"].mean()
            ),
            "transmission_source_coverage": float(
                feature_table["transmission_source_available_flag"].mean()
            ),
            "ecp_source_coverage": float(feature_table["ecp_source_available_flag"].mean()),
            "missing_rate_by_feature": {
                column: float(feature_table[column].isna().mean())
                for column in feature_table.columns
                if pd.api.types.is_numeric_dtype(feature_table[column])
            },
        }
    )
    return feature_table, _dictionary(feature_table), checks


def _ablation_model() -> Pipeline:
    """Return the fixed residual model used on both sides of the ablation."""

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
    train_residual = (
        train["dispatch_down_mwh"]
        - train["dispatch_down_mwh_latest_observed"]
    )
    model.fit(train[columns], train_residual)
    prediction = np.maximum(
        score["dispatch_down_mwh_latest_observed"].to_numpy(dtype=float)
        + model.predict(score[columns]),
        0.0,
    )
    return float(mean_absolute_error(score["dispatch_down_mwh"], prediction))


def build_outage_ablation_report(
    model_data: pd.DataFrame,
    feature_table: pd.DataFrame,
    feature_artifact_path: Path | str,
) -> dict[str, object]:
    """Compare identical models with and without Issue #6 feature columns.

    Selection uses three expanding-window folds inside the 70% training period,
    plus the following 15% validation period.  The last 15% remains sealed.
    """

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
    if additions.duplicated(keys).any():
        raise DataQualityError("Ablation feature rows contain duplicate modelling keys")
    overlap = baseline.merge(additions, on=keys, how="inner", validate="one_to_one")
    if len(overlap) != len(baseline):
        raise DataQualityError(
            "Outage feature ablation requires one feature row per baseline row"
        )

    excluded = IDENTIFIER_COLUMNS | TARGET_COLUMNS
    baseline_columns = [
        column
        for column in baseline.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(baseline[column])
    ]
    candidate_columns = [
        column
        for column in additions.columns
        if column not in keys
        and pd.api.types.is_numeric_dtype(additions[column])
        and additions[column].notna().any()
    ]
    if not candidate_columns:
        raise DataQualityError("No usable numeric outage/constraint features for ablation")

    unique_times = np.array(sorted(overlap["issue_timestamp_utc"].unique()))
    if len(unique_times) < 20:
        raise DataQualityError("Ablation requires at least 20 unique issue timestamps")
    train_end = int(len(unique_times) * 0.70)
    validation_end = int(len(unique_times) * 0.85)
    train_times = unique_times[:train_end]
    validation_times = unique_times[train_end:validation_end]
    development = overlap.loc[
        overlap["issue_timestamp_utc"].isin(unique_times[:validation_end])
    ].copy()

    fold_specs: list[tuple[str, np.ndarray, np.ndarray]] = []
    for index, (fit_fraction, score_fraction) in enumerate(
        ((0.40, 0.60), (0.60, 0.80), (0.80, 1.00)),
        start=1,
    ):
        fit_end = max(1, int(len(train_times) * fit_fraction))
        score_end = max(fit_end + 1, int(len(train_times) * score_fraction))
        fold_specs.append(
            (
                f"train_rolling_{index}",
                train_times[:fit_end],
                train_times[fit_end:score_end],
            )
        )
    fold_specs.append(("validation", train_times, validation_times))

    candidate_full_columns = baseline_columns + candidate_columns
    fold_metrics: list[dict[str, object]] = []
    for name, fit_times, score_times in fold_specs:
        fit = development.loc[development["issue_timestamp_utc"].isin(fit_times)]
        score = development.loc[development["issue_timestamp_utc"].isin(score_times)]
        baseline_mae = _score_ablation_model(fit, score, baseline_columns)
        candidate_mae = _score_ablation_model(fit, score, candidate_full_columns)
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
    mean_improvement = float(improvements.mean())
    include = bool(mean_improvement >= 0.15 and improvements.min() >= 0.0)
    reason = (
        "Candidate improved mean development MAE by at least 15% without degrading any fold."
        if include
        else "Candidate did not clear the 15% mean-MAE gate with no degrading development fold."
    )
    artifact_path = Path(feature_artifact_path)
    return {
        "experiment_name": "outage_constraint_signals",
        "feature_group": "generation_transmission_outage_and_ecp_constraint_pressure",
        "feature_artifact": artifact_path.as_posix(),
        "feature_artifact_sha256": _sha256(artifact_path),
        "overlap_rows": int(len(overlap)),
        "development_rows": int(len(development)),
        "development_fold_count": len(fold_metrics),
        "candidate_feature_count": len(candidate_columns),
        "candidate_features": candidate_columns,
        "model": "hist_gradient_boosting_dispatch_residual",
        "selection_gate": {
            "mean_mae_improvement_min": 0.15,
            "maximum_fold_degradation": 0.0,
        },
        "metrics": {
            "folds": fold_metrics,
            "mean_baseline_mae_mwh": float(
                np.mean([metric["baseline_mae_mwh"] for metric in fold_metrics])
            ),
            "mean_candidate_mae_mwh": float(
                np.mean([metric["candidate_mae_mwh"] for metric in fold_metrics])
            ),
            "mean_mae_improvement_fraction": mean_improvement,
            "worst_fold_improvement_fraction": float(improvements.min()),
        },
        "status": "accepted" if include else "rejected_development_gate",
        "production_decision": "include" if include else "exclude",
        "reason": reason,
        "final_test_accessed": False,
    }
