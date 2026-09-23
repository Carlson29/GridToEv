"""Forecast-vintage parsers and leakage-safe as-of joins."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

import pandas as pd

from .eirgrid_history import DataQualityError


VINTAGE_KEY = [
    "feature_name",
    "target_timestamp_utc",
    "published_at_utc",
    "revision",
]


def _utc_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
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
    value = value.removeprefix("I_")
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def parse_semo_forecast_xml(
    source: bytes | Path,
    *,
    source_id: str,
    source_url: str,
    downloaded_at: str | pd.Timestamp,
    revision: int = 1,
) -> pd.DataFrame:
    """Parse supported SEMO forecast XML into one long vintage table.

    SEMO report timestamps are published without an explicit offset. They are
    treated as UTC, while the original strings are retained for auditability.
    """

    payload = source.read_bytes() if isinstance(source, Path) else source
    root = ET.fromstring(payload)
    dataset = root.attrib.get("DatasetName", "")
    published_original = root.attrib.get("PublishTime")
    if not published_original:
        raise DataQualityError(f"{source_id} has no XML PublishTime")
    published_at = _utc_timestamp(published_original)
    downloaded = _utc_timestamp(downloaded_at)
    if downloaded < published_at:
        raise DataQualityError(f"{source_id} was downloaded before its publication time")

    rows: list[dict[str, object]] = []

    def append_value(
        element: ET.Element,
        *,
        value_field: str,
        feature_name: str,
        region: str,
        resource_name: str | None = None,
    ) -> None:
        start_original = _content(element, "StartTime")
        end_original = _content(element, "EndTime")
        raw_value = _content(element, value_field)
        if start_original is None or raw_value in {None, ""}:
            return
        rows.append(
            {
                "feature_name": feature_name,
                "value": float(raw_value),
                "unit": "MW",
                "region": region,
                "resource_name": resource_name,
                "target_timestamp_utc": _utc_timestamp(start_original),
                "target_end_timestamp_utc": (
                    _utc_timestamp(end_original) if end_original else pd.NaT
                ),
                "target_time_original": start_original,
                "published_at_utc": published_at,
                "published_time_original": published_original,
                "downloaded_at_utc": downloaded,
                "revision": revision,
                "vintage_id": f"{source_id}:{published_at.isoformat()}",
                "source_id": source_id,
                "source_url": source_url,
                "dataset_name": dataset,
            }
        )

    if dataset == "PUB_15MinAggWindFcst":
        for element in root.findall("PUB_15MinAggWindFcst"):
            append_value(
                element,
                value_field="Forecast",
                feature_name="wind_forecast_mw_all",
                region="ALL",
            )
    elif dataset == "PUB_DailyLoadFcst":
        fields = {
            "LoadForecastROI": ("demand_forecast_mw_roi", "ROI"),
            "LoadForecastNI": ("demand_forecast_mw_ni", "NI"),
            "AggregatedForecast": ("demand_forecast_mw_all", "ALL"),
        }
        for element in root.findall("PUB_DailyLoadFcst"):
            for field, (feature_name, region) in fields.items():
                append_value(
                    element,
                    value_field=field,
                    feature_name=feature_name,
                    region=region,
                )
    elif dataset == "PUB_DailyIntconNTC":
        for element in root.findall("PUB_DailyIntconNTC"):
            resource_name = _content(element, "ResourceName") or "unknown"
            slug = _resource_slug(resource_name)
            append_value(
                element,
                value_field="MaximumImportMW",
                feature_name=f"interconnector_max_import_mw_{slug}",
                region="ALL",
                resource_name=resource_name,
            )
            append_value(
                element,
                value_field="MaximumExportMW",
                feature_name=f"interconnector_max_export_mw_{slug}",
                region="ALL",
                resource_name=resource_name,
            )
    else:
        raise DataQualityError(f"Unsupported SEMO forecast dataset: {dataset}")

    if not rows:
        raise DataQualityError(f"{source_id} produced no forecast rows")
    return pd.DataFrame(rows).sort_values(
        ["target_timestamp_utc"], kind="stable"
    ).reset_index(drop=True)


def parse_smartgrid_snapshot(
    payload: bytes | str | dict[str, object],
    *,
    area: str,
    source_id: str,
    source_url: str,
    downloaded_at: str | pd.Timestamp,
) -> pd.DataFrame:
    """Parse a wind/solar dashboard snapshot using retrieval as safe availability.

    The dashboard does not expose historic publication timestamps. Setting
    ``published_at_utc`` to retrieval time is conservative: a snapshot can only
    be selected by a model issue time after this pipeline actually observed it.
    """

    if isinstance(payload, bytes):
        document = json.loads(payload)
    elif isinstance(payload, str):
        document = json.loads(payload)
    else:
        document = payload
    downloaded = _utc_timestamp(downloaded_at)
    rows: list[dict[str, object]] = []
    forecast_field = f"{area.upper()}_FCAST"
    for source_row in document.get("Rows", []):  # type: ignore[union-attr]
        if "MW_FORECAST" in source_row:
            raw_value = source_row.get("MW_FORECAST")
        elif source_row.get("FieldName") == forecast_field:
            raw_value = source_row.get("Value")
        else:
            continue
        original = source_row.get("EffectiveTime") or source_row.get("Effective_Date")
        if original is None or raw_value in {None, ""}:
            continue
        region = str(source_row.get("Region", "ALL")).upper()
        target_timestamp = pd.to_datetime(original, dayfirst=True, utc=True, errors="raise")
        rows.append(
            {
                "feature_name": f"{area.lower()}_forecast_mw_{region.lower()}",
                "value": float(raw_value),
                "unit": "MW",
                "region": region,
                "resource_name": None,
                "target_timestamp_utc": target_timestamp,
                "target_end_timestamp_utc": pd.NaT,
                "target_time_original": original,
                "published_at_utc": downloaded,
                "published_time_original": None,
                "downloaded_at_utc": downloaded,
                "revision": 1,
                "vintage_id": f"{source_id}:{downloaded.isoformat()}",
                "source_id": source_id,
                "source_url": source_url,
                "dataset_name": f"smartgrid_{area}_snapshot",
            }
        )
    return pd.DataFrame(rows)


def deduplicate_vintages(vintages: pd.DataFrame) -> pd.DataFrame:
    if vintages.empty:
        return vintages.copy()
    missing = sorted(set(VINTAGE_KEY + ["downloaded_at_utc"]) - set(vintages.columns))
    if missing:
        raise DataQualityError(f"Forecast vintages missing columns: {missing}")
    result = vintages.copy()
    for column in (
        "target_timestamp_utc",
        "published_at_utc",
        "downloaded_at_utc",
    ):
        result[column] = pd.to_datetime(result[column], utc=True, errors="raise")
    result = result.sort_values(
        [*VINTAGE_KEY, "downloaded_at_utc"], kind="stable"
    ).drop_duplicates(VINTAGE_KEY, keep="last")
    return result.reset_index(drop=True)


def asof_join_forecasts(
    modelling_rows: pd.DataFrame,
    vintages: pd.DataFrame,
    *,
    feature_names: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Attach the latest forecast that was published by each model issue time."""

    required_model = {"issue_timestamp_utc", "target_timestamp_utc"}
    missing_model = sorted(required_model - set(modelling_rows.columns))
    if missing_model:
        raise DataQualityError(f"Modelling rows missing columns: {missing_model}")
    result = modelling_rows.copy().reset_index(drop=True)
    result["issue_timestamp_utc"] = pd.to_datetime(
        result["issue_timestamp_utc"], utc=True, errors="raise"
    )
    result["target_timestamp_utc"] = pd.to_datetime(
        result["target_timestamp_utc"], utc=True, errors="raise"
    )
    clean = deduplicate_vintages(vintages)
    selected_features = (
        sorted(clean["feature_name"].dropna().unique())
        if feature_names is None
        else list(feature_names)
    )
    result["_row_id"] = range(len(result))

    for feature_name in selected_features:
        feature = clean.loc[clean["feature_name"].eq(feature_name)].copy()
        candidates = result[
            ["_row_id", "issue_timestamp_utc", "target_timestamp_utc"]
        ].merge(feature, on="target_timestamp_utc", how="left")
        candidates = candidates.loc[
            candidates["published_at_utc"].notna()
            & (candidates["published_at_utc"] <= candidates["issue_timestamp_utc"])
        ]
        candidates = candidates.sort_values(
            [
                "_row_id",
                "published_at_utc",
                "revision",
                "downloaded_at_utc",
            ],
            kind="stable",
        ).drop_duplicates("_row_id", keep="last")
        selected = candidates.set_index("_row_id")
        result[feature_name] = result["_row_id"].map(selected["value"].to_dict())
        published_values = result["_row_id"].map(
            selected["published_at_utc"].astype("object").to_dict()
        )
        result[f"{feature_name}_published_at_utc"] = pd.to_datetime(
            published_values, utc=True
        )
        source_values = (
            selected["source_id"].to_dict() if "source_id" in selected else {}
        )
        result[f"{feature_name}_source_id"] = result["_row_id"].map(source_values)
        result[f"{feature_name}_age_minutes"] = (
            result["issue_timestamp_utc"]
            - result[f"{feature_name}_published_at_utc"]
        ).dt.total_seconds() / 60
        result[f"{feature_name}_available_flag"] = (
            result[feature_name].notna().astype("int8")
        )

    result = result.drop(columns="_row_id")
    publication_columns = [
        column for column in result if column.endswith("_published_at_utc")
    ]
    for column in publication_columns:
        leakage = result[column].notna() & (
            result[column] > result["issue_timestamp_utc"]
        )
        if leakage.any():
            raise DataQualityError(f"As-of join leaked {int(leakage.sum())} rows via {column}")
    return result


def forecast_quality_report(vintages: pd.DataFrame) -> dict[str, object]:
    clean = deduplicate_vintages(vintages)
    coverage: dict[str, dict[str, object]] = {}
    for feature_name, feature in clean.groupby("feature_name", sort=True):
        lead_minutes = (
            feature["target_timestamp_utc"] - feature["published_at_utc"]
        ).dt.total_seconds() / 60
        retrieval_lag_minutes = (
            feature["downloaded_at_utc"] - feature["published_at_utc"]
        ).dt.total_seconds() / 60
        coverage[feature_name] = {
            "rows": int(len(feature)),
            "vintages": int(feature["vintage_id"].nunique())
            if "vintage_id" in feature
            else int(feature["published_at_utc"].nunique()),
            "first_target_utc": feature["target_timestamp_utc"].min().isoformat(),
            "last_target_utc": feature["target_timestamp_utc"].max().isoformat(),
            "first_publication_utc": feature["published_at_utc"].min().isoformat(),
            "last_publication_utc": feature["published_at_utc"].max().isoformat(),
            "forecast_lead_minutes_p50": float(lead_minutes.quantile(0.5)),
            "forecast_lead_minutes_p95": float(lead_minutes.quantile(0.95)),
            "retrieval_lag_minutes_p50": float(retrieval_lag_minutes.quantile(0.5)),
            "retrieval_lag_minutes_p95": float(retrieval_lag_minutes.quantile(0.95)),
        }
    return {
        "rows": int(len(clean)),
        "duplicate_vintage_keys": int(clean.duplicated(VINTAGE_KEY).sum()),
        "feature_coverage": coverage,
    }
