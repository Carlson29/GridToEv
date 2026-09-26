"""Leakage-safe forecast-error and grid-headroom feature engineering.

Every feature in this module is evaluated from the point of view of
``issue_timestamp_utc``. Forecasts must already have been published and actual
measurements must have completed their half-hour settlement interval before a
row is allowed to use them.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .eirgrid_history import DataQualityError
from .forecast_vintages import asof_join_forecasts, deduplicate_vintages


ACTUAL_MAP = {
    "demand": (
        "demand_forecast_mw_all",
        "eirgrid_all_island_demand_mw",
    ),
    "wind": (
        "wind_forecast_mw_all",
        "eirgrid_all_island_wind_generation_mw",
    ),
    "solar": (
        "solar_forecast_mw_all",
        "eirgrid_all_island_solar_generation_mw",
    ),
}

INTERCONNECTORS = {
    "nimoyle": "eirgrid_moyle_flow_mw",
    "roiewic": "eirgrid_ewic_flow_mw",
    "roigrlk": "eirgrid_greenlink_flow_mw",
}

STATE_COLUMNS = {
    "eirgrid_all_island_demand_mw": "latest_completed_demand_mw",
    "eirgrid_all_island_wind_generation_mw": "latest_completed_wind_generation_mw",
    "eirgrid_all_island_solar_generation_mw": "latest_completed_solar_generation_mw",
    "eirgrid_ewic_flow_mw": "latest_completed_ewic_flow_mw",
    "eirgrid_moyle_flow_mw": "latest_completed_moyle_flow_mw",
    "eirgrid_greenlink_flow_mw": "latest_completed_greenlink_flow_mw",
    "eirgrid_snsp_ratio": "latest_completed_snsp_ratio",
}

LABEL_COLUMNS = (
    "dispatch_down_mwh",
    "curtailment_mwh",
    "constraint_mwh",
    "dispatch_down_available_flag",
)

ROLLING_WINDOWS = {
    "2h": "2h",
    "6h": "6h",
    "12h": "12h",
    "24h": "24h",
}

STALE_AFTER_MINUTES = {
    "demand_forecast_mw_all": 360,
    "wind_forecast_mw_all": 120,
    "solar_forecast_mw_all": 1440,
}

STATE_STALE_AFTER_MINUTES = 90


def _utc(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column in result:
            result[column] = pd.to_datetime(result[column], utc=True, errors="raise")
    return result


def _forecast_staleness(result: pd.DataFrame) -> None:
    """Add explicit age-limit flags for every forecast family used here."""

    thresholds = dict(STALE_AFTER_MINUTES)
    for connector in INTERCONNECTORS:
        thresholds[f"interconnector_max_import_mw_{connector}"] = 1440
        thresholds[f"interconnector_max_export_mw_{connector}"] = 1440
    for feature, threshold in thresholds.items():
        age_column = f"{feature}_age_minutes"
        if age_column not in result:
            result[age_column] = np.nan
        result[f"{feature}_stale_flag"] = (
            result[age_column].isna() | result[age_column].gt(threshold)
        ).astype("int8")


def _attach_latest_completed_state(
    result: pd.DataFrame,
    history: pd.DataFrame,
) -> pd.DataFrame:
    available = [column for column in STATE_COLUMNS if column in history]
    state = history[["timestamp_utc", *available]].copy()
    state["latest_completed_observed_at_utc"] = (
        state["timestamp_utc"] + pd.Timedelta(minutes=30)
    )
    state = state.drop(columns="timestamp_utc").rename(columns=STATE_COLUMNS)
    state = state.sort_values("latest_completed_observed_at_utc")

    left = result.copy()
    left["_row_id"] = range(len(left))
    left = left.sort_values("issue_timestamp_utc")
    merged = pd.merge_asof(
        left,
        state,
        left_on="issue_timestamp_utc",
        right_on="latest_completed_observed_at_utc",
        direction="backward",
        allow_exact_matches=True,
    )
    merged["latest_completed_state_age_minutes"] = (
        merged["issue_timestamp_utc"]
        - merged["latest_completed_observed_at_utc"]
    ).dt.total_seconds() / 60
    merged["latest_completed_state_stale_flag"] = (
        merged["latest_completed_state_age_minutes"].isna()
        | merged["latest_completed_state_age_minutes"].gt(
            STATE_STALE_AFTER_MINUTES
        )
    ).astype("int8")
    return merged.sort_values("_row_id").drop(columns="_row_id").reset_index(drop=True)


def _attach_target_labels(result: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    label_columns = [column for column in LABEL_COLUMNS if column in history]
    labels = history[["timestamp_utc", *label_columns]].rename(
        columns={"timestamp_utc": "target_timestamp_utc"}
    )
    # Existing as-of matrices contain a label-availability flag. Recompute it
    # from the authoritative history instead of producing merge suffixes.
    drop_existing = [column for column in label_columns if column in result]
    if drop_existing:
        result = result.drop(columns=drop_existing)
    result = result.merge(labels, on="target_timestamp_utc", how="left", validate="many_to_one")
    if "dispatch_down_available_flag" in result:
        result["dispatch_down_available_flag"] = (
            result["dispatch_down_available_flag"].fillna(0).astype("int8")
        )
        result["history_label_available_flag"] = (
            result["dispatch_down_available_flag"]
        )
    return result


def _engineer_system_state(result: pd.DataFrame) -> None:
    wind = result.get("wind_forecast_mw_all", pd.Series(np.nan, index=result.index))
    solar = result.get("solar_forecast_mw_all", pd.Series(np.nan, index=result.index))
    demand = result.get("demand_forecast_mw_all", pd.Series(np.nan, index=result.index))
    result["forecast_variable_renewable_mw"] = pd.concat([wind, solar], axis=1).sum(
        axis=1, min_count=2
    )
    result["forecast_net_load_mw"] = demand - result["forecast_variable_renewable_mw"]
    valid_demand = demand.where(demand.gt(0))
    result["forecast_renewable_share_ratio"] = (
        result["forecast_variable_renewable_mw"] / valid_demand
    )
    result["forecast_renewable_surplus_mw"] = (
        result["forecast_variable_renewable_mw"] - demand
    ).clip(lower=0)

    completed_flows: list[str] = []
    headroom_columns: list[str] = []
    utilisation_numerators: list[pd.Series] = []
    utilisation_denominators: list[pd.Series] = []
    fresh_state = result["latest_completed_state_stale_flag"].eq(0)
    for connector, history_column in INTERCONNECTORS.items():
        flow_column = STATE_COLUMNS[history_column]
        completed_flows.append(flow_column)
        flow = result.get(flow_column, pd.Series(np.nan, index=result.index)).where(
            fresh_state
        ).abs()
        export_column = f"interconnector_max_export_mw_{connector}"
        import_column = f"interconnector_max_import_mw_{connector}"
        export_capacity = result.get(export_column, pd.Series(np.nan, index=result.index))
        import_capacity = result.get(import_column, pd.Series(np.nan, index=result.index))
        headroom_column = f"{connector}_export_headroom_mw"
        result[headroom_column] = (export_capacity - flow).clip(lower=0)
        headroom_columns.append(headroom_column)
        capacity = pd.concat([export_capacity, import_capacity], axis=1).max(
            axis=1, skipna=False
        )
        utilisation_numerators.append(flow)
        utilisation_denominators.append(capacity)
        result[f"{connector}_utilisation_ratio"] = flow / capacity.where(capacity.gt(0))

    flow_frame = result[completed_flows]
    result["latest_completed_net_interconnector_flow_mw"] = flow_frame.sum(
        axis=1, min_count=len(completed_flows)
    )
    import_contribution = result["latest_completed_net_interconnector_flow_mw"].clip(
        lower=0
    )
    result["forecast_snsp_proxy_ratio"] = (
        result["forecast_variable_renewable_mw"] + import_contribution
    ) / valid_demand
    result["interconnector_export_headroom_mw"] = result[headroom_columns].sum(
        axis=1, min_count=len(headroom_columns)
    )
    total_flow = pd.concat(utilisation_numerators, axis=1).sum(
        axis=1, min_count=len(utilisation_numerators)
    )
    total_capacity = pd.concat(utilisation_denominators, axis=1).sum(
        axis=1, min_count=len(utilisation_denominators)
    )
    result["interconnector_utilisation_ratio"] = total_flow / total_capacity.where(
        total_capacity.gt(0)
    )


def _attach_revision_features(
    result: pd.DataFrame,
    vintages: pd.DataFrame,
) -> None:
    clean = deduplicate_vintages(vintages)
    keys = result[["issue_timestamp_utc", "target_timestamp_utc"]].copy()
    keys["_row_id"] = range(len(keys))
    for short_name, (feature_name, _) in ACTUAL_MAP.items():
        feature = clean.loc[clean["feature_name"].eq(feature_name)].copy()
        candidates = keys.merge(feature, on="target_timestamp_utc", how="left")
        candidates = candidates.loc[
            candidates["published_at_utc"].notna()
            & (candidates["published_at_utc"] <= candidates["issue_timestamp_utc"])
        ].sort_values(
            ["_row_id", "published_at_utc", "revision", "downloaded_at_utc"],
            kind="stable",
        )
        known_values = candidates.groupby("_row_id", sort=False)["value"].apply(list)
        values_by_row = known_values.to_dict()
        result[f"{short_name}_forecast_known_vintage_count"] = pd.Series(
            [len(values_by_row.get(index, [])) for index in result.index],
            index=result.index,
            dtype="int16",
        )
        result[f"{short_name}_forecast_vintage_disagreement_std_mw"] = [
            float(np.std(values)) if values else np.nan
            for index in result.index
            for values in [values_by_row.get(index, [])]
        ]
        result[f"{short_name}_forecast_vintage_range_mw"] = [
            float(max(values) - min(values)) if values else np.nan
            for index in result.index
            for values in [values_by_row.get(index, [])]
        ]
        result[f"{short_name}_forecast_revision_delta_mw"] = [
            float(values[-1] - values[-2]) if len(values) >= 2 else np.nan
            for index in result.index
            for values in [values_by_row.get(index, [])]
        ]


def _historical_error_scores(
    history: pd.DataFrame,
    vintages: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    clean = deduplicate_vintages(vintages)
    vintage_targets = pd.DatetimeIndex(clean["target_timestamp_utc"].dropna().unique())
    actuals = history.loc[history["timestamp_utc"].isin(vintage_targets)].copy()
    if actuals.empty:
        return pd.DataFrame()
    rows = pd.DataFrame({"target_timestamp_utc": actuals["timestamp_utc"]})
    rows["issue_timestamp_utc"] = rows["target_timestamp_utc"] - pd.Timedelta(
        minutes=horizon
    )
    rows["forecast_horizon_minutes"] = horizon
    joined = asof_join_forecasts(
        rows,
        clean,
        feature_names=[forecast for forecast, _ in ACTUAL_MAP.values()],
    )
    actual_columns = [actual for _, actual in ACTUAL_MAP.values() if actual in actuals]
    joined = joined.merge(
        actuals[["timestamp_utc", *actual_columns]].rename(
            columns={"timestamp_utc": "target_timestamp_utc"}
        ),
        on="target_timestamp_utc",
        how="left",
        validate="one_to_one",
    )
    joined["observed_at_utc"] = joined["target_timestamp_utc"] + pd.Timedelta(minutes=30)
    return joined


def _attach_error_features(
    result: pd.DataFrame,
    history: pd.DataFrame,
    vintages: pd.DataFrame,
) -> None:
    for horizon in sorted(result["forecast_horizon_minutes"].dropna().unique()):
        horizon = int(horizon)
        scores = _historical_error_scores(history, vintages, horizon)
        current_index = result.index[result["forecast_horizon_minutes"].eq(horizon)]
        for short_name, (forecast_column, actual_column) in ACTUAL_MAP.items():
            prefix = f"{short_name}_forecast_error"
            output_columns = [
                f"{prefix}_mw_latest_completed",
                f"{prefix}_observed_at_utc",
                f"{prefix}_forecast_published_at_utc",
                f"{prefix}_age_minutes",
            ]
            for label in ROLLING_WINDOWS:
                output_columns.extend(
                    [
                        f"{prefix}_mae_{label}",
                        f"{prefix}_bias_{label}",
                        f"{prefix}_count_{label}",
                    ]
                )
            for column in output_columns:
                if column not in result:
                    if column.endswith("_utc"):
                        result[column] = pd.Series(
                            pd.NaT,
                            index=result.index,
                            dtype="datetime64[ns, UTC]",
                        )
                    else:
                        result[column] = np.nan
            if (
                scores.empty
                or forecast_column not in scores
                or actual_column not in scores
                or current_index.empty
            ):
                continue

            score = scores[
                [
                    "observed_at_utc",
                    forecast_column,
                    actual_column,
                    f"{forecast_column}_published_at_utc",
                ]
            ].copy()
            score["error"] = score[actual_column] - score[forecast_column]
            score = score.loc[
                score["error"].notna()
                & score[f"{forecast_column}_published_at_utc"].notna()
            ].sort_values("observed_at_utc")
            if score.empty:
                continue
            for label, window in ROLLING_WINDOWS.items():
                rolling = score.set_index("observed_at_utc")["error"].rolling(
                    window, closed="right"
                )
                score[f"mae_{label}"] = rolling.apply(
                    lambda values: float(np.abs(values).mean()), raw=True
                ).to_numpy()
                score[f"bias_{label}"] = rolling.mean().to_numpy()
                score[f"count_{label}"] = rolling.count().to_numpy()

            left = result.loc[current_index, ["issue_timestamp_utc"]].copy()
            left["_row_id"] = current_index
            left = left.sort_values("issue_timestamp_utc")
            merged = pd.merge_asof(
                left,
                score,
                left_on="issue_timestamp_utc",
                right_on="observed_at_utc",
                direction="backward",
                allow_exact_matches=True,
            ).set_index("_row_id")
            result.loc[merged.index, f"{prefix}_mw_latest_completed"] = merged["error"]
            result.loc[merged.index, f"{prefix}_observed_at_utc"] = merged[
                "observed_at_utc"
            ]
            result.loc[merged.index, f"{prefix}_forecast_published_at_utc"] = merged[
                f"{forecast_column}_published_at_utc"
            ]
            result.loc[merged.index, f"{prefix}_age_minutes"] = (
                merged["issue_timestamp_utc"] - merged["observed_at_utc"]
            ).dt.total_seconds() / 60
            for label in ROLLING_WINDOWS:
                for statistic in ("mae", "bias", "count"):
                    result.loc[
                        merged.index, f"{prefix}_{statistic}_{label}"
                    ] = merged[f"{statistic}_{label}"]


def _add_missingness_flags(result: pd.DataFrame) -> pd.DataFrame:
    """Pair nullable model inputs with an explicit, numeric missingness signal."""

    flags: dict[str, pd.Series] = {}
    candidates = list(result.columns)
    for column in candidates:
        if column.endswith(("_missing_flag", "_available_flag", "_stale_flag")):
            continue
        if not result[column].isna().any():
            continue
        if pd.api.types.is_numeric_dtype(result[column]) or column.endswith("_utc"):
            flags[f"{column}_missing_flag"] = result[column].isna().astype("int8")
    if not flags:
        return result
    return pd.concat([result, pd.DataFrame(flags, index=result.index)], axis=1)


def _dictionary_for(table: pd.DataFrame) -> pd.DataFrame:
    exact_formulas = {
        "forecast_variable_renewable_mw": "wind_forecast_mw_all + solar_forecast_mw_all",
        "forecast_net_load_mw": "demand_forecast_mw_all - forecast_variable_renewable_mw",
        "forecast_renewable_share_ratio": "forecast_variable_renewable_mw / demand_forecast_mw_all",
        "forecast_renewable_surplus_mw": "max(forecast_variable_renewable_mw - demand_forecast_mw_all, 0)",
        "forecast_snsp_proxy_ratio": "(forecast variable renewables + positive latest completed net interconnector flow) / demand forecast",
        "interconnector_export_headroom_mw": "sum(max(export NTC - abs(latest completed flow), 0)) across Moyle, EWIC and Greenlink",
        "interconnector_utilisation_ratio": "sum(abs(latest completed flow)) / sum(max(import NTC, export NTC))",
    }
    rows: list[dict[str, object]] = []
    for column in table:
        if column in {"issue_timestamp_utc", "target_timestamp_utc", "forecast_horizon_minutes"}:
            role = "identifier"
        elif column in LABEL_COLUMNS or column == "history_label_available_flag":
            role = "target_or_label_availability"
        elif column.endswith(("_flag", "_count_2h", "_count_6h", "_count_12h", "_count_24h")):
            role = "quality_or_availability"
        elif column.endswith(("_source_id", "_published_at_utc", "_observed_at_utc")):
            role = "provenance"
        else:
            role = "model_feature"

        if column.endswith("_flag") or "_count_" in column or column.endswith("_count"):
            unit = "integer"
        elif column.endswith(("_mw", "_mwh")) or "_mw_" in column:
            unit = "MW" if not column.endswith("_mwh") else "MWh"
        elif "forecast_error_mae_" in column or "forecast_error_bias_" in column:
            unit = "MW"
        elif column.endswith("_ratio"):
            unit = "ratio"
        elif column.endswith("_minutes"):
            unit = "minutes"
        elif column.endswith("_utc"):
            unit = "UTC timestamp"
        else:
            unit = "source-defined"

        if column.endswith("_missing_flag"):
            formula = (
                f"1 when {column.removesuffix('_missing_flag')} is missing, "
                "otherwise 0"
            )
            source = "derived quality flag"
        elif column.endswith("_stale_flag"):
            formula = (
                "1 when the input is unavailable or older than its documented "
                "threshold"
            )
            source = "derived from publication or observation age"
        elif column in exact_formulas:
            formula = exact_formulas[column]
            source = "derived from leakage-safe forecast and completed system-state inputs"
        elif "forecast_error" in column:
            formula = "actual - as-of forecast; rolling statistics use only errors observed by issue time"
            source = "SEMO/EirGrid forecast vintages plus EirGrid actuals"
        elif "forecast_revision" in column or "vintage_" in column:
            formula = "summary of forecast vintages published no later than issue_timestamp_utc"
            source = "SEMO/EirGrid forecast vintage archive"
        elif column.startswith("latest_completed_"):
            formula = "latest half-hour observation whose interval ended no later than issue_timestamp_utc"
            source = "EirGrid historical system data"
        elif column.endswith("_export_headroom_mw"):
            formula = "max(export NTC - abs(latest completed flow), 0); null when state is stale"
            source = "SEMO NTC forecast plus EirGrid completed interconnector flow"
        elif column.endswith("_utilisation_ratio"):
            formula = "abs(latest completed flow) / max(import NTC, export NTC); null when state is stale"
            source = "SEMO NTC forecast plus EirGrid completed interconnector flow"
        else:
            formula = "carried through from its named source field"
            source = "as-of forecast matrix or EirGrid history"

        if column.endswith("_missing_flag"):
            availability = "calculated from the paired field on the same row"
        elif column.endswith("_stale_flag"):
            availability = "calculated from age known at issue time"
        elif "forecast_error" in column:
            availability = "error observation time <= issue time; forecast publication time <= its historical issue time"
        elif "revision" in column or "vintage" in column:
            availability = "only revisions published at or before issue time"
        elif column.startswith("latest_completed_"):
            availability = "observation interval end <= issue time"
        elif column.endswith("_published_at_utc"):
            availability = "timestamp must be <= issue time"
        else:
            availability = "available at issue time or explicitly marked missing"

        rows.append(
            {
                "column": column,
                "role": role,
                "unit": unit,
                "formula": formula,
                "source": source,
                "availability_rule": availability,
                "missingness_handling": (
                    "not applicable" if column.endswith("_missing_flag") else "retain null and use paired missing/availability flag; fitted pipeline median-imputes numeric inputs"
                ),
                "dtype": str(table[column].dtype),
            }
        )
    return pd.DataFrame(rows)


def validate_forecast_feature_table(table: pd.DataFrame) -> dict[str, object]:
    """Validate natural keys, horizon alignment, and all availability cut-offs."""

    required = {
        "issue_timestamp_utc",
        "target_timestamp_utc",
        "forecast_horizon_minutes",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise DataQualityError(f"Forecast feature table missing columns: {missing}")
    checked = _utc(table, ["issue_timestamp_utc", "target_timestamp_utc"])
    duplicate_keys = int(
        checked.duplicated(["issue_timestamp_utc", "forecast_horizon_minutes"]).sum()
    )
    if duplicate_keys:
        raise DataQualityError(f"Forecast feature table has {duplicate_keys} duplicate natural keys")
    expected_target = checked["issue_timestamp_utc"] + pd.to_timedelta(
        checked["forecast_horizon_minutes"], unit="m"
    )
    horizon_violations = int((checked["target_timestamp_utc"] != expected_target).sum())
    if horizon_violations:
        raise DataQualityError(f"Forecast feature table has {horizon_violations} horizon alignment violations")

    future_publications = 0
    future_observations = 0
    for column in checked:
        if column.endswith("_published_at_utc"):
            values = pd.to_datetime(checked[column], utc=True, errors="coerce")
            future_publications += int(
                (values.notna() & values.gt(checked["issue_timestamp_utc"])).sum()
            )
        elif column.endswith("_observed_at_utc"):
            values = pd.to_datetime(checked[column], utc=True, errors="coerce")
            future_observations += int(
                (values.notna() & values.gt(checked["issue_timestamp_utc"])).sum()
            )
    if future_publications:
        raise DataQualityError(f"Forecast feature table has {future_publications} future publication leaks")
    if future_observations:
        raise DataQualityError(f"Forecast feature table has {future_observations} future observation leaks")

    numeric = checked.select_dtypes(include=[np.number])
    nonfinite = int(np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).sum())
    if nonfinite:
        raise DataQualityError(f"Forecast feature table has {nonfinite} infinite numeric values")
    unhandled = [
        column
        for column in numeric
        if numeric[column].isna().any() and f"{column}_missing_flag" not in checked
    ]
    if unhandled:
        raise DataQualityError(f"Missing numeric features have no explicit flag: {unhandled}")
    return {
        "duplicate_natural_keys": duplicate_keys,
        "horizon_alignment_violations": horizon_violations,
        "future_publication_violations": future_publications,
        "future_observation_violations": future_observations,
        "nonfinite_numeric_cells": nonfinite,
        "unhandled_missing_feature_columns": unhandled,
    }


def build_forecast_feature_table(
    asof: pd.DataFrame,
    history: pd.DataFrame,
    vintages: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Build the Issue #4 model table, dictionary, and quality summary."""

    result = _utc(
        asof,
        [
            "issue_timestamp_utc",
            "target_timestamp_utc",
            *[column for column in asof if column.endswith("_published_at_utc")],
        ],
    ).reset_index(drop=True)
    history = _utc(history, ["timestamp_utc"])
    # Fail before engineering anything if a selected input revision is from the future.
    validate_seed = result.copy()
    validate_seed = _add_missingness_flags(validate_seed)
    validate_forecast_feature_table(validate_seed)

    result = _attach_latest_completed_state(result, history)
    result = _attach_target_labels(result, history)
    _forecast_staleness(result)
    _engineer_system_state(result)
    _attach_revision_features(result, vintages)
    _attach_error_features(result, history, vintages)
    result = _add_missingness_flags(result)
    result = result.sort_values(
        ["issue_timestamp_utc", "forecast_horizon_minutes"], kind="stable"
    ).reset_index(drop=True)

    checks = validate_forecast_feature_table(result)
    availability_columns = [column for column in result if column.endswith("_available_flag")]
    missing_columns = [column for column in result if column.endswith("_missing_flag")]
    checks.update(
        {
            "rows": int(len(result)),
            "columns": int(len(result.columns)),
            "forecast_horizons_minutes": sorted(
                int(value) for value in result["forecast_horizon_minutes"].unique()
            ),
            "first_issue_utc": result["issue_timestamp_utc"].min().isoformat(),
            "last_issue_utc": result["issue_timestamp_utc"].max().isoformat(),
            "label_rows": int(result.get("history_label_available_flag", 0).sum()),
            "availability_rate_by_feature": {
                column.removesuffix("_available_flag"): float(result[column].mean())
                for column in availability_columns
            },
            "missing_rate_by_feature": {
                column.removesuffix("_missing_flag"): float(result[column].mean())
                for column in missing_columns
            },
            "staleness_threshold_minutes": {
                **STALE_AFTER_MINUTES,
                "interconnector_capacity": 1440,
                "latest_completed_system_state": STATE_STALE_AFTER_MINUTES,
            },
            "known_coverage_limitation": (
                "Forecast errors and completed-state features remain null when the retained "
                "forecast archive and published actual-history period do not overlap; no values "
                "are backfilled or fabricated."
            ),
        }
    )
    return result, _dictionary_for(result), checks
