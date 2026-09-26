"""SEMO market and operational signals with publication-time-safe joins."""

from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from .eirgrid_history import DataQualityError


SCHEMA_VERSION = "semo-market-v1"

REPORT_NAMES = {
    "PUB_15MinAggWindFcst": "Aggregated Wind Forecast",
    "PUB_4DayAggRollWindUnitFcst": "Four Day Aggregated Rolling Wind Unit Forecast",
    "PUB_DailyLoadFcst": "Daily Load Forecast Summary",
    "PUB_RTCOperationalSchedule": "RTIC OPERATIONAL SCHEDULE REPORT",
    "PUB_HlfHrlyAggregatedPN": "Aggregated Final Physical Notifications",
    "PUB_HrlyForecastImbalance": "Forecast Imbalance",
    "PUB_HrlyNetImbalVolumeForecast": "Net Imbalance Volume Forecast",
    "PUB_DailyIntconNTC": "Daily Interconnector NTC",
    "PUB_5MinImbalPrc": "Imbalance Price Report (Imbalance Pricing Period)",
    "PUB_30MinAvgImbalPrc": "Imbalance Price Report (Imbalance Settlement Period)",
    "PUB_HrlySsiiSiff": (
        "System Shortfall Imbalance Index and System Imbalance Flattening Factor"
    ),
}

SUPPORTED_DATASETS = tuple(REPORT_NAMES)

# source XML field -> (normalized feature, unit, jurisdiction)
FIELD_SPECS = {
    "PUB_15MinAggWindFcst": {
        "Forecast": ("wind_forecast_15min_mw_all", "MW", "ALL"),
    },
    "PUB_4DayAggRollWindUnitFcst": {
        "LoadForecastROI": ("wind_forecast_4day_mw_roi", "MW", "ROI"),
        "LoadForecastNI": ("wind_forecast_4day_mw_ni", "MW", "NI"),
        "AggregatedForecast": ("wind_forecast_4day_mw_all", "MW", "ALL"),
    },
    "PUB_DailyLoadFcst": {
        "LoadForecastROI": ("demand_forecast_mw_roi", "MW", "ROI"),
        "LoadForecastNI": ("demand_forecast_mw_ni", "MW", "NI"),
        "AggregatedForecast": ("demand_forecast_mw_all", "MW", "ALL"),
    },
    "PUB_RTCOperationalSchedule": {
        "ScheduledQuantity": ("rtc_scheduled_quantity_mw", "MW", "ALL"),
    },
    "PUB_HlfHrlyAggregatedPN": {
        "StartMWAggregatedPN": ("aggregated_pn_start_mw", "MW", "ALL"),
        "EndMWAggregatedPN": ("aggregated_pn_end_mw", "MW", "ALL"),
    },
    "PUB_HrlyForecastImbalance": {
        "TotalPN": ("forecast_total_pn_mw", "MW", "ALL"),
        "NetInterconnectorSchedule": (
            "forecast_net_interconnector_schedule_mw",
            "MW",
            "ALL",
        ),
        "TSODemandForecast": ("forecast_tso_demand_mw", "MW", "ALL"),
        "TSORenewableForecastPD": (
            "forecast_renewable_dispatchable_mw",
            "MW",
            "ALL",
        ),
        "TSORenewableForecastNPDR": (
            "forecast_renewable_nondispatchable_mw",
            "MW",
            "ALL",
        ),
        "CalculatedImbalance": (
            "forecast_calculated_imbalance_mw",
            "MW",
            "ALL",
        ),
    },
    "PUB_HrlyNetImbalVolumeForecast": {
        "NetImbalanceForecast": (
            "net_imbalance_volume_forecast_mwh",
            "MWh",
            "ALL",
        ),
    },
    "PUB_DailyIntconNTC": {
        "MaximumImportMW": ("interconnector_max_import_mw", "MW", "ALL"),
        "MaximumExportMW": ("interconnector_max_export_mw", "MW", "ALL"),
    },
    "PUB_5MinImbalPrc": {
        "NetImbalanceVolume": ("net_imbalance_volume_5min_mwh", "MWh", "ALL"),
        "ImbalancePrice": (
            "imbalance_price_5min_eur_per_mwh",
            "EUR/MWh",
            "ALL",
        ),
        "TotalUnitAvailability": ("total_unit_availability_mw", "MW", "ALL"),
        "ShortTermReserveQuantity": (
            "short_term_reserve_quantity_mw",
            "MW",
            "ALL",
        ),
        "OperatingReserveRequirement": (
            "operating_reserve_requirement_mw",
            "MW",
            "ALL",
        ),
    },
    "PUB_30MinAvgImbalPrc": {
        "NetImbalanceVolume": ("net_imbalance_volume_30min_mwh", "MWh", "ALL"),
        "ImbalanceSettlementPrice": (
            "imbalance_price_30min_eur_per_mwh",
            "EUR/MWh",
            "ALL",
        ),
    },
    "PUB_HrlySsiiSiff": {
        "Ssii": ("system_shortfall_imbalance_index", "ratio", "ALL"),
        "Siff": ("system_imbalance_flattening_factor", "ratio", "ALL"),
    },
}

ACTUAL_OUTCOME_DATASETS = {
    "PUB_5MinImbalPrc",
    "PUB_30MinAvgImbalPrc",
    "PUB_HrlySsiiSiff",
}

PUBLICATION_SCOPED_DATASETS = {
    "PUB_HrlyNetImbalVolumeForecast",
    "PUB_HrlySsiiSiff",
}

SNAPSHOT_OUTPUT_NAMES = {
    "net_imbalance_volume_forecast_mwh": "net_imbalance_volume_forecast_mwh",
    "net_imbalance_volume_5min_mwh": "lagged_net_imbalance_volume_5min_mwh",
    "imbalance_price_5min_eur_per_mwh": (
        "lagged_imbalance_price_5min_eur_per_mwh"
    ),
    "total_unit_availability_mw": "lagged_total_unit_availability_mw",
    "short_term_reserve_quantity_mw": "lagged_short_term_reserve_quantity_mw",
    "operating_reserve_requirement_mw": (
        "lagged_operating_reserve_requirement_mw"
    ),
    "net_imbalance_volume_30min_mwh": "lagged_net_imbalance_volume_mwh",
    "imbalance_price_30min_eur_per_mwh": (
        "lagged_imbalance_price_eur_per_mwh"
    ),
    "system_shortfall_imbalance_index": "lagged_system_shortfall_imbalance_index",
    "system_imbalance_flattening_factor": (
        "lagged_system_imbalance_flattening_factor"
    ),
}

STALE_AFTER_MINUTES = {
    "rtc_scheduled_generation_mw": 60,
    "rtc_interconnector_schedule_mw": 60,
    "rtc_total_scheduled_quantity_mw": 60,
    "aggregated_pn_start_mw": 120,
    "aggregated_pn_end_mw": 120,
    "forecast_calculated_imbalance_mw": 120,
    "net_imbalance_volume_forecast_mwh": 120,
    "lagged_imbalance_price_eur_per_mwh": 90,
    "lagged_net_imbalance_volume_mwh": 90,
    "lagged_imbalance_price_5min_eur_per_mwh": 45,
    "lagged_net_imbalance_volume_5min_mwh": 45,
}

INTERCONNECTOR_FLOW_COLUMNS = {
    "nimoyle": "eirgrid_moyle_flow_mw",
    "roiewic": "eirgrid_ewic_flow_mw",
    "roigrlk": "eirgrid_greenlink_flow_mw",
}


def _utc_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _content(element: ET.Element, name: str) -> str | None:
    if name in element.attrib:
        return element.attrib[name]
    child = element.find(name)
    return child.text if child is not None else None


def _resource_slug(value: str) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        value.removeprefix("I_").lower(),
    ).strip("_")


def parse_semo_market_xml(
    source: bytes | Path,
    *,
    source_id: str,
    source_url: str,
    downloaded_at: str | pd.Timestamp,
    catalog_published_at: str | pd.Timestamp | None = None,
    revision: int = 1,
) -> pd.DataFrame:
    """Parse one supported SEMO XML report into an auditable long table."""

    payload = source.read_bytes() if isinstance(source, Path) else source
    root = ET.fromstring(payload)
    dataset = str(root.attrib.get("DatasetName", ""))
    if dataset not in FIELD_SPECS:
        raise DataQualityError(f"Unsupported SEMO market dataset: {dataset}")
    root_published_raw = root.attrib.get("PublishTime")
    if not root_published_raw and catalog_published_at is None:
        raise DataQualityError(f"{source_id} has no publication timestamp")
    publication_candidates = [
        _utc_timestamp(value)
        for value in (root_published_raw, catalog_published_at)
        if value is not None
    ]
    downloaded = _utc_timestamp(downloaded_at)
    rows: list[dict[str, object]] = []

    for element in root.findall(dataset):
        row_publication = _content(element, "PublishTime")
        published = max(
            [
                *publication_candidates,
                *([_utc_timestamp(row_publication)] if row_publication else []),
            ]
        )
        if downloaded < published:
            raise DataQualityError(
                f"{source_id} was downloaded before its effective publication time"
            )
        start_raw = _content(element, "StartTime")
        end_raw = _content(element, "EndTime")
        start = _utc_timestamp(start_raw) if start_raw else pd.NaT
        end = _utc_timestamp(end_raw) if end_raw else pd.NaT
        resource_name = _content(element, "ResourceName")
        resource_type = _content(element, "ResourceType")
        for source_field, (feature_name, unit, jurisdiction) in FIELD_SPECS[
            dataset
        ].items():
            raw_value = _content(element, source_field)
            effective_source_field = source_field
            if (
                raw_value in {None, ""}
                and dataset == "PUB_5MinImbalPrc"
                and source_field == "ImbalancePrice"
                and _content(element, "DefaultPriceUsage") == "Y"
            ):
                raw_value = _content(element, "MarketBackupPrice")
                effective_source_field = "MarketBackupPrice"
            if raw_value in {None, ""}:
                continue
            normalized_name = feature_name
            if dataset == "PUB_DailyIntconNTC":
                if not resource_name:
                    raise DataQualityError(f"{source_id} NTC row has no resource name")
                normalized_name = f"{feature_name}_{_resource_slug(resource_name)}"
            rows.append(
                {
                    "report_name": REPORT_NAMES[dataset],
                    "dataset_name": dataset,
                    "feature_name": normalized_name,
                    "value": float(raw_value),
                    "unit": unit,
                    "jurisdiction": jurisdiction,
                    "resource_name": resource_name,
                    "resource_type": resource_type,
                    "target_timestamp_utc": start,
                    "target_end_timestamp_utc": end,
                    "published_at_utc": published,
                    "downloaded_at_utc": downloaded,
                    "revision": revision,
                    "source_id": source_id,
                    "source_url": source_url,
                    "schema_version": SCHEMA_VERSION,
                    "schema_variant": (
                        "attributes"
                        if effective_source_field in element.attrib
                        else "child_elements"
                    ),
                    "actual_outcome_flag": int(dataset in ACTUAL_OUTCOME_DATASETS),
                }
            )
    if not rows:
        raise DataQualityError(f"{source_id} produced no supported market-signal rows")
    result = pd.DataFrame(rows)
    for column in (
        "target_timestamp_utc",
        "target_end_timestamp_utc",
        "published_at_utc",
        "downloaded_at_utc",
    ):
        result[column] = pd.to_datetime(result[column], utc=True)
    actual = result["actual_outcome_flag"].eq(1) & result[
        "target_end_timestamp_utc"
    ].notna()
    early = actual & (
        result["published_at_utc"] < result["target_end_timestamp_utc"]
    )
    if early.any():
        raise DataQualityError(
            f"{source_id} publishes {int(early.sum())} actual outcomes before interval end"
        )
    return result.sort_values(
        ["published_at_utc", "target_timestamp_utc", "feature_name"],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)


def deduplicate_market_signals(signals: pd.DataFrame) -> pd.DataFrame:
    required = {
        "dataset_name",
        "feature_name",
        "published_at_utc",
        "downloaded_at_utc",
        "revision",
        "value",
    }
    missing = sorted(required - set(signals.columns))
    if missing:
        raise DataQualityError(f"SEMO market signals missing columns: {missing}")
    result = signals.copy()
    for column in (
        "target_timestamp_utc",
        "target_end_timestamp_utc",
        "published_at_utc",
        "downloaded_at_utc",
    ):
        if column in result:
            result[column] = pd.to_datetime(result[column], utc=True, errors="coerce")
    key = [
        "dataset_name",
        "feature_name",
        "resource_name",
        "target_timestamp_utc",
        "published_at_utc",
        "revision",
    ]
    key = [column for column in key if column in result]
    return (
        result.sort_values([*key, "downloaded_at_utc"], kind="stable")
        .drop_duplicates(key, keep="last")
        .reset_index(drop=True)
    )


def _expanded_target_rows(signals: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    target_signals = signals.loc[
        signals["target_timestamp_utc"].notna()
        & signals["actual_outcome_flag"].eq(0)
        & ~signals["feature_name"].isin(SNAPSHOT_OUTPUT_NAMES)
    ]
    for source in target_signals.to_dict("records"):
        start = pd.Timestamp(source["target_timestamp_utc"])
        end = source.get("target_end_timestamp_utc")
        bucket = start.floor("30min")
        buckets = [bucket]
        if pd.notna(end):
            end_timestamp = pd.Timestamp(end)
            buckets = []
            while bucket < end_timestamp:
                buckets.append(bucket)
                bucket += pd.Timedelta(minutes=30)
        for target in buckets:
            rows.append({**source, "target_timestamp_utc": target})
    if not rows:
        return pd.DataFrame()
    expanded = pd.DataFrame(rows)
    rtc = expanded.loc[expanded["feature_name"].eq("rtc_scheduled_quantity_mw")]
    other = expanded.loc[~expanded["feature_name"].eq("rtc_scheduled_quantity_mw")]
    outputs: list[pd.DataFrame] = []
    group_columns = [
        "report_name",
        "dataset_name",
        "feature_name",
        "target_timestamp_utc",
        "published_at_utc",
    ]
    if not other.empty:
        outputs.append(
            other.groupby(group_columns, dropna=False, as_index=False).agg(
                value=("value", "mean"),
                downloaded_at_utc=("downloaded_at_utc", "max"),
            )
        )
    if not rtc.empty:
        by_resource = rtc.groupby(
            [
                "report_name",
                "dataset_name",
                "target_timestamp_utc",
                "published_at_utc",
                "resource_name",
            ],
            dropna=False,
            as_index=False,
        ).agg(value=("value", "mean"), downloaded_at_utc=("downloaded_at_utc", "max"))
        connector = by_resource["resource_name"].fillna("").str.startswith("I_")
        for feature_name, subset in (
            ("rtc_scheduled_generation_mw", by_resource.loc[~connector]),
            ("rtc_interconnector_schedule_mw", by_resource.loc[connector]),
            ("rtc_total_scheduled_quantity_mw", by_resource),
        ):
            if subset.empty:
                continue
            grouped = subset.groupby(
                [
                    "report_name",
                    "dataset_name",
                    "target_timestamp_utc",
                    "published_at_utc",
                ],
                as_index=False,
            ).agg(value=("value", "sum"), downloaded_at_utc=("downloaded_at_utc", "max"))
            grouped["feature_name"] = feature_name
            outputs.append(grouped)
    return pd.concat(outputs, ignore_index=True, sort=False) if outputs else pd.DataFrame()


def _attach_target_features(result: pd.DataFrame, signals: pd.DataFrame) -> None:
    prepared = _expanded_target_rows(signals)
    if prepared.empty:
        return
    keys = result[
        ["issue_timestamp_utc", "target_timestamp_utc"]
    ].copy()
    keys["_row_id"] = range(len(keys))
    for feature_name, feature in prepared.groupby("feature_name", sort=True):
        candidates = keys.merge(feature, on="target_timestamp_utc", how="left")
        candidates = candidates.loc[
            candidates["published_at_utc"].notna()
            & candidates["published_at_utc"].le(candidates["issue_timestamp_utc"])
        ].sort_values(
            ["_row_id", "published_at_utc", "downloaded_at_utc"],
            kind="stable",
        ).drop_duplicates("_row_id", keep="last")
        selected = candidates.set_index("_row_id")
        result[feature_name] = result.index.map(selected["value"].to_dict())
        published = result.index.map(
            selected["published_at_utc"].astype("object").to_dict()
        )
        result[f"{feature_name}_published_at_utc"] = pd.to_datetime(
            published,
            utc=True,
        )
        result[f"{feature_name}_age_minutes"] = (
            result["issue_timestamp_utc"]
            - result[f"{feature_name}_published_at_utc"]
        ).dt.total_seconds() / 60
        result[f"{feature_name}_available_flag"] = (
            result[feature_name].notna().astype("int8")
        )
        threshold = STALE_AFTER_MINUTES.get(feature_name, 1440)
        result[f"{feature_name}_stale_flag"] = (
            result[feature_name].isna()
            | result[f"{feature_name}_age_minutes"].gt(threshold)
        ).astype("int8")


def _attach_snapshot_features(result: pd.DataFrame, signals: pd.DataFrame) -> None:
    for source_name, output_name in SNAPSHOT_OUTPUT_NAMES.items():
        feature = signals.loc[signals["feature_name"].eq(source_name)].copy()
        if feature.empty:
            continue
        feature = feature.sort_values(
            ["published_at_utc", "downloaded_at_utc"], kind="stable"
        ).drop_duplicates("published_at_utc", keep="last")
        columns = [
            "published_at_utc",
            "downloaded_at_utc",
            "target_end_timestamp_utc",
            "value",
        ]
        left = result[["issue_timestamp_utc"]].copy()
        left["_row_id"] = range(len(left))
        merged = pd.merge_asof(
            left.sort_values("issue_timestamp_utc"),
            feature[columns].sort_values("published_at_utc"),
            left_on="issue_timestamp_utc",
            right_on="published_at_utc",
            direction="backward",
            allow_exact_matches=True,
        ).set_index("_row_id").sort_index()
        result[output_name] = merged["value"]
        result[f"{output_name}_published_at_utc"] = merged["published_at_utc"]
        result[f"{output_name}_observed_at_utc"] = merged[
            "target_end_timestamp_utc"
        ]
        result[f"{output_name}_age_minutes"] = (
            result["issue_timestamp_utc"]
            - result[f"{output_name}_published_at_utc"]
        ).dt.total_seconds() / 60
        result[f"{output_name}_available_flag"] = (
            result[output_name].notna().astype("int8")
        )
        threshold = STALE_AFTER_MINUTES.get(output_name, 120)
        result[f"{output_name}_stale_flag"] = (
            result[output_name].isna()
            | result[f"{output_name}_age_minutes"].gt(threshold)
        ).astype("int8")


def _attach_completed_state(result: pd.DataFrame, history: pd.DataFrame) -> None:
    state_columns = [
        "eirgrid_all_island_generation_mw",
        *INTERCONNECTOR_FLOW_COLUMNS.values(),
    ]
    available = [column for column in state_columns if column in history]
    state = history[["timestamp_utc", *available]].copy()
    state["semo_state_observed_at_utc"] = state["timestamp_utc"] + pd.Timedelta(
        minutes=30
    )
    state = state.drop(columns="timestamp_utc").sort_values(
        "semo_state_observed_at_utc"
    )
    left = result.copy()
    left["_row_id"] = range(len(left))
    merged = pd.merge_asof(
        left.sort_values("issue_timestamp_utc"),
        state,
        left_on="issue_timestamp_utc",
        right_on="semo_state_observed_at_utc",
        direction="backward",
        allow_exact_matches=True,
    ).sort_values("_row_id")
    for column in [*available, "semo_state_observed_at_utc"]:
        result[column] = merged[column].to_numpy()
    result["semo_state_age_minutes"] = (
        result["issue_timestamp_utc"] - result["semo_state_observed_at_utc"]
    ).dt.total_seconds() / 60
    result["semo_state_stale_flag"] = (
        result["semo_state_age_minutes"].isna()
        | result["semo_state_age_minutes"].gt(90)
    ).astype("int8")


def _engineer_features(result: pd.DataFrame) -> None:
    pn_start = result.get("aggregated_pn_start_mw")
    pn_end = result.get("aggregated_pn_end_mw")
    if pn_start is not None and pn_end is not None:
        result["aggregated_pn_average_mw"] = (pn_start + pn_end) / 2

    fresh_state = result["semo_state_stale_flag"].eq(0)
    schedule = result.get(
        "rtc_scheduled_generation_mw",
        pd.Series(np.nan, index=result.index),
    )
    actual_generation = result.get(
        "eirgrid_all_island_generation_mw",
        pd.Series(np.nan, index=result.index),
    ).where(fresh_state)
    result["scheduled_minus_actual_generation_mw"] = schedule - actual_generation

    connector_schedule = result.get(
        "rtc_interconnector_schedule_mw",
        pd.Series(np.nan, index=result.index),
    )
    demand = result.get(
        "demand_forecast_mw_all",
        pd.Series(np.nan, index=result.index),
    )
    result["downward_margin_proxy_mw"] = (
        schedule + connector_schedule.clip(lower=0) - demand
    ).clip(lower=0)

    if "forecast_calculated_imbalance_mw" in result:
        result["forecast_imbalance_mw"] = result[
            "forecast_calculated_imbalance_mw"
        ]
    reserve = result.get("lagged_short_term_reserve_quantity_mw")
    requirement = result.get("lagged_operating_reserve_requirement_mw")
    if reserve is not None and requirement is not None:
        result["lagged_reserve_surplus_mw"] = reserve - requirement

    headroom_columns: list[str] = []
    for connector, flow_column in INTERCONNECTOR_FLOW_COLUMNS.items():
        capacity_column = f"interconnector_max_export_mw_{connector}"
        if capacity_column not in result:
            continue
        flow = result.get(flow_column, pd.Series(np.nan, index=result.index)).where(
            fresh_state
        )
        output = f"{connector}_export_headroom_mw"
        result[output] = (result[capacity_column] - flow.abs()).clip(lower=0)
        headroom_columns.append(output)
    if headroom_columns:
        result["interconnector_export_headroom_mw"] = result[
            headroom_columns
        ].sum(axis=1, min_count=len(INTERCONNECTOR_FLOW_COLUMNS))


def _add_missing_flags(result: pd.DataFrame) -> pd.DataFrame:
    flags: dict[str, pd.Series] = {}
    for column in list(result.columns):
        if column.endswith(("_flag", "_source_id")):
            continue
        if result[column].isna().any() and (
            pd.api.types.is_numeric_dtype(result[column]) or column.endswith("_utc")
        ):
            flags[f"{column}_missing_flag"] = result[column].isna().astype("int8")
    return (
        pd.concat([result, pd.DataFrame(flags, index=result.index)], axis=1)
        if flags
        else result
    )


def validate_semo_feature_matrix(table: pd.DataFrame) -> dict[str, int]:
    required = {
        "issue_timestamp_utc",
        "target_timestamp_utc",
        "forecast_horizon_minutes",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise DataQualityError(f"SEMO feature matrix missing columns: {missing}")
    checked = table.copy()
    for column in [
        "issue_timestamp_utc",
        "target_timestamp_utc",
        *[
            name
            for name in checked
            if name.endswith(("_published_at_utc", "_observed_at_utc"))
        ],
    ]:
        checked[column] = pd.to_datetime(checked[column], utc=True, errors="coerce")
    duplicates = int(
        checked.duplicated(["issue_timestamp_utc", "forecast_horizon_minutes"]).sum()
    )
    if duplicates:
        raise DataQualityError(f"SEMO feature matrix has {duplicates} duplicate keys")
    expected = checked["issue_timestamp_utc"] + pd.to_timedelta(
        checked["forecast_horizon_minutes"], unit="m"
    )
    horizon_violations = int(checked["target_timestamp_utc"].ne(expected).sum())
    if horizon_violations:
        raise DataQualityError(
            f"SEMO feature matrix has {horizon_violations} horizon violations"
        )
    future_publications = sum(
        int(
            (
                checked[column].notna()
                & checked[column].gt(checked["issue_timestamp_utc"])
            ).sum()
        )
        for column in checked
        if column.endswith("_published_at_utc")
    )
    future_observations = sum(
        int(
            (
                checked[column].notna()
                & checked[column].gt(checked["issue_timestamp_utc"])
            ).sum()
        )
        for column in checked
        if column.endswith("_observed_at_utc")
    )
    if future_publications:
        raise DataQualityError(
            f"SEMO feature matrix has {future_publications} future publications"
        )
    if future_observations:
        raise DataQualityError(
            f"SEMO feature matrix has {future_observations} future observations"
        )
    numeric = checked.select_dtypes(include=[np.number])
    infinite = int(np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).sum())
    if infinite:
        raise DataQualityError(f"SEMO feature matrix has {infinite} infinite values")
    return {
        "duplicate_natural_keys": duplicates,
        "horizon_alignment_violations": horizon_violations,
        "future_publication_violations": future_publications,
        "future_observation_violations": future_observations,
        "nonfinite_numeric_cells": infinite,
    }


def _dictionary(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in table:
        if column in {
            "issue_timestamp_utc",
            "target_timestamp_utc",
            "forecast_horizon_minutes",
        }:
            role = "identifier"
        elif column.endswith("_published_at_utc") or column.endswith(
            "_observed_at_utc"
        ):
            role = "provenance"
        elif column.endswith("_flag"):
            role = "availability_or_quality"
        else:
            role = "model_feature"
        if column.endswith("_flag"):
            unit = "integer"
        elif "eur_per_mwh" in column:
            unit = "EUR/MWh"
        elif "_mwh" in column:
            unit = "MWh"
        elif "_mw" in column:
            unit = "MW"
        elif column.endswith("_ratio") or "index" in column or "factor" in column:
            unit = "ratio"
        elif column.endswith("_minutes"):
            unit = "minutes"
        elif column.endswith("_utc"):
            unit = "UTC timestamp"
        else:
            unit = "source-defined"
        if column == "scheduled_minus_actual_generation_mw":
            formula = "target RTC scheduled generation - latest completed actual generation"
        elif column == "downward_margin_proxy_mw":
            formula = "max(RTC generation + positive interconnector schedule - demand forecast, 0)"
        elif column == "interconnector_export_headroom_mw":
            formula = "sum(max(export NTC - abs(latest completed flow), 0))"
        elif column == "lagged_reserve_surplus_mw":
            formula = "short-term reserve quantity - operating reserve requirement"
        elif column.endswith("_missing_flag"):
            formula = f"1 when {column.removesuffix('_missing_flag')} is missing"
        elif column.endswith("_stale_flag"):
            formula = "1 when missing or older than the documented threshold"
        else:
            formula = "normalized SEMO field or its as-of availability metadata"
        rows.append(
            {
                "column": column,
                "role": role,
                "unit": unit,
                "formula": formula,
                "source": "SEMO static reports; completed-state fields use EirGrid history",
                "availability_rule": (
                    "publication and any actual interval end must be <= issue_timestamp_utc"
                ),
                "missingness_handling": (
                    "retain null and use paired missing/availability/staleness flag"
                ),
                "dtype": str(table[column].dtype),
            }
        )
    return pd.DataFrame(rows)


def build_semo_feature_matrix(
    modelling_rows: pd.DataFrame,
    signals: pd.DataFrame,
    history: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Build target-aligned forecasts and lagged market outcomes for model rows."""

    result = modelling_rows[
        ["issue_timestamp_utc", "target_timestamp_utc", "forecast_horizon_minutes"]
    ].copy()
    result["issue_timestamp_utc"] = pd.to_datetime(
        result["issue_timestamp_utc"], utc=True, errors="raise"
    )
    result["target_timestamp_utc"] = pd.to_datetime(
        result["target_timestamp_utc"], utc=True, errors="raise"
    )
    history = history.copy()
    history["timestamp_utc"] = pd.to_datetime(
        history["timestamp_utc"], utc=True, errors="raise"
    )
    clean = deduplicate_market_signals(signals)
    actual = clean["actual_outcome_flag"].eq(1) & clean[
        "target_end_timestamp_utc"
    ].notna()
    early = actual & clean["published_at_utc"].lt(
        clean["target_end_timestamp_utc"]
    )
    if early.any():
        raise DataQualityError(
            f"SEMO signals contain {int(early.sum())} prematurely available outcomes"
        )

    _attach_target_features(result, clean)
    result = result.copy()
    _attach_snapshot_features(result, clean)
    result = result.copy()
    _attach_completed_state(result, history)
    result = result.copy()
    _engineer_features(result)
    result = _add_missing_flags(result)
    result = result.sort_values(
        ["issue_timestamp_utc", "forecast_horizon_minutes"], kind="stable"
    ).reset_index(drop=True)
    checks = validate_semo_feature_matrix(result)

    report_coverage: dict[str, dict[str, object]] = {}
    for dataset, report_name in REPORT_NAMES.items():
        rows = clean.loc[clean["dataset_name"].eq(dataset)]
        report_coverage[dataset] = {
            "report_name": report_name,
            "rows": int(len(rows)),
            "first_publication_utc": (
                rows["published_at_utc"].min().isoformat() if not rows.empty else None
            ),
            "last_publication_utc": (
                rows["published_at_utc"].max().isoformat() if not rows.empty else None
            ),
            "features": sorted(rows["feature_name"].unique()) if not rows.empty else [],
        }
    availability = [column for column in result if column.endswith("_available_flag")]
    stale = [column for column in result if column.endswith("_stale_flag")]
    checks.update(
        {
            "signal_rows": int(len(clean)),
            "feature_rows": int(len(result)),
            "feature_columns": int(len(result.columns)),
            "first_issue_utc": result["issue_timestamp_utc"].min().isoformat(),
            "last_issue_utc": result["issue_timestamp_utc"].max().isoformat(),
            "report_coverage": report_coverage,
            "availability_rate_by_feature": {
                column.removesuffix("_available_flag"): float(result[column].mean())
                for column in availability
            },
            "stale_rate_by_feature": {
                column.removesuffix("_stale_flag"): float(result[column].mean())
                for column in stale
            },
            "missing_rate_by_feature": {
                column: float(result[column].isna().mean())
                for column in result
                if pd.api.types.is_numeric_dtype(result[column])
            },
        }
    )
    return result, _dictionary(result), checks


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_semo_ablation_record(
    baseline: pd.DataFrame,
    market_features: pd.DataFrame,
    signal_path: Path,
) -> dict[str, object]:
    """Record the named SEMO ablation, including an honest no-overlap rejection."""

    baseline_issue = pd.to_datetime(
        baseline["issue_timestamp_utc"], utc=True, errors="raise"
    )
    market_issue = pd.to_datetime(
        market_features["issue_timestamp_utc"], utc=True, errors="raise"
    )
    overlap_rows = int(baseline_issue.isin(set(market_issue)).sum())
    if overlap_rows:
        status = "ready_for_leakage_safe_benchmark"
        decision = "holdout_pending"
        reason = "Temporal overlap exists; run the frozen rolling benchmark before inclusion."
    else:
        status = "rejected_no_temporal_overlap"
        decision = "exclude"
        reason = (
            "The retained SEMO archive does not overlap the frozen labelled benchmark, "
            "so performance cannot be estimated without fabricating historical vintages."
        )
    return {
        "schema_version": 1,
        "experiment_name": "semo_market_operational_signals",
        "feature_group": "semo_imbalance_schedule_pn_ntc",
        "status": status,
        "production_decision": decision,
        "reason": reason,
        "overlap_rows": overlap_rows,
        "baseline_rows": int(len(baseline)),
        "market_feature_rows": int(len(market_features)),
        "baseline_first_issue_utc": baseline_issue.min().isoformat(),
        "baseline_last_issue_utc": baseline_issue.max().isoformat(),
        "market_first_issue_utc": market_issue.min().isoformat(),
        "market_last_issue_utc": market_issue.max().isoformat(),
        "signal_dataset_sha256": _sha256(signal_path),
        "metrics": None,
        "final_test_accessed": False,
    }
