from __future__ import annotations

import os
import secrets
import logging
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator

from .actuals import ActualsService
from .constants import DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH
from .daily_curtailment import DEFAULT_HISTORY
from .daily_model import DEFAULT_REPORT, EARLIEST_TARGET_DATE, DailyCurtailmentService
from .inference import DatasetSelectionError, FeatureValidationError, PredictionService
from .raw_examples import v1_raw_example, v2_raw_example
from .raw_prediction import (
    V1_RAW_CURRENT_FIELDS, V1_RAW_HISTORY_FIELDS, V1_RATIO_FIELDS,
    V1_SIGNED_FIELDS, build_v1_features_from_raw, build_v2_features_from_raw,
)
from .release_guard import DEFAULT_REPORT_PATH


ForecastHorizon = Literal[30, 60]
logger = logging.getLogger(__name__)

TAG_SERVICE = "Service"
TAG_V1 = "V1 — 30/60-minute model"
TAG_V2 = "V2 — daily curtailment model"
TAG_ACTUALS = "Observed outcomes"

API_DESCRIPTION = """## Choose the right model

| Model | Predicts | Input | Start here |
| --- | --- | --- | --- |
| **V1: 30/60-minute** | Renewable **dispatch-down** risk and MWh at one target half-hour, 30 or 60 minutes after an issue time. | A historical dataset issue timestamp; complete original source observations with 48 half-hours of history; or an already engineered feature snapshot. | `GET /model-info` for estimators/scores, `GET /model-info/v1/raw-input-schema` for original-value inputs, then `POST /predict/v1/from-raw` or `POST /predict/from-dataset`. |
| **V2: daily curtailment (experimental)** | Whether **curtailment** occurs and total curtailment MWh over a full UTC calendar day. This is a different target and time scale from v1. | One historical/current UTC date, 96 user-supplied hourly forecasts for a target day, or a future start date and 1–7 complete days. | `GET /model-info/daily-curtailment` for estimators/scores, `GET /model-info/daily-curtailment/raw-input-schema` for original-value inputs, then `POST /predict/curtailment/from-raw`, `/day` or `/window`. |

V1's dataset replay endpoints use a **fixed historical dataset**. In particular, `/predict/latest` means the latest *dataset row*, **not the current time**. A V1 window is repeated 30/60-minute predictions for at most 24 historical hours, **not a day-ahead forecast**. V2's `/day` route fetches archived 24-hour-lead weather for dates through today UTC. Its `/window` route fetches **live GFS forecasts** for tomorrow through seven days ahead and returns one prediction per full UTC day. Multi-day lead accuracy has **not been validated** against the historical V2 test metric. The version shown in the Swagger header does not choose a model; each prediction reports its own `model_version`.

Send the shared `X-API-Key` using **Authorize** for protected routes. `GET /health` is public and shows whether optional v2 loaded. To compare a prediction with reality, copy `target_timestamp_utc` from v1 to `/actuals/v1`, or `target_date_utc` from v2 to `/actuals/daily-curtailment`. Actuals come from a separately refreshed EirGrid snapshot and may be `pending` or `missing`.

`GET /models/catalog` lists the two tasks. Model-info scores are historical evaluations, **not the error of an individual prediction**. Raw-input routes validate shape and claimed availability times but cannot independently verify that caller-supplied values were truly published at those times; they do not fetch a live V1 source feed.

The prefilled **raw-input** request bodies demonstrate the **next target beyond each model dataset**, without adding that target or its outcome to the dataset. V1 uses the latest bundled source row to predict the next unrecorded half-hour; its example publication times are illustrative because the dataset does not preserve release receipts. V2 uses archived Open-Meteo day-ahead forecasts for the first day after its model dataset. These are frozen-dataset demos, **not forecasts for today's clock time**. Replace values and timestamps with genuinely available inputs for any new prediction. If you do not have all 49 V1 observations or 96 V2 forecasts, use the simpler dataset/weather-backed endpoints instead.
"""

OPENAPI_TAGS = [
    {"name": TAG_SERVICE, "description": "Public service readiness. V1 readiness and optional V2 availability are reported separately."},
    {"name": TAG_V1, "description": "V1 predicts 30- or 60-minute-ahead renewable dispatch-down from a historical half-hour dataset or a full feature snapshot."},
    {"name": TAG_V2, "description": "Experimental V2 predicts curtailment event probability and total MWh for a whole UTC day."},
    {"name": TAG_ACTUALS, "description": "Observed EirGrid outcomes for V1 half-hours and V2 complete UTC days; these are not model predictions."},
]


def _require_timezone(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(
            "A timezone is required; use ISO 8601 UTC format such as "
            "2026-01-15T12:30:00Z"
        )
    return value.astimezone(timezone.utc)


def _v1_raw_field_description(name: str) -> str:
    unit = "0–1 fraction" if name in V1_RATIO_FIELDS else (
        "EUR/MWh" if name == "entsoe_price_eur_mwh" else "MW"
    )
    if name.startswith("entsoe_"):
        label = name.removeprefix("entsoe_").replace("_generation_mw", " generation")
        label = label.replace("actual_load_mw", "actual electricity load")
        label = label.replace("price_eur_mwh", "electricity price")
        source = "ENTSO-E"
    else:
        label = name.removeprefix("eirgrid_").replace("all_island", "all-island (Ireland and Northern Ireland)")
        label = label.replace("ie_", "Ireland ").replace("ewic", "East-West interconnector")
        label = label.replace("greenlink", "Greenlink interconnector")
        label = label.replace("snsp", "system non-synchronous penetration")
        label = label.replace("_mw", "").replace("_ratio", "")
        source = "EirGrid"
    label = label.replace("_", " ")
    return f"{source} original source value: {label} ({unit}) for the completed issue half-hour. Not a future target."


V1RawCurrentObservation = create_model(
    "V1RawCurrentObservation",
    __config__=ConfigDict(extra="forbid"),
    **{
        name: (
            float,
            Field(
                description=_v1_raw_field_description(name),
                **({} if name in V1_SIGNED_FIELDS else {"ge": 0}),
                **({"le": 1} if name in V1_RATIO_FIELDS else {}),
            ),
        )
        for name in V1_RAW_CURRENT_FIELDS
    },
    observed_dispatch_down_mwh=(float, Field(
        ge=0, description="Observed dispatch-down MWh in the completed issue half-hour, already published by issue time. This is a PAST input, not the future dispatch-down target.",
    )),
    available_at_utc=(datetime, Field(
        description="Latest UTC publication/availability time across all current source values, including observed dispatch-down. Must be no later than issue_timestamp_utc; user-provided provenance cannot be independently verified.",
    )),
)


class V1RawHistoryObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp_utc: datetime = Field(description="Completed UTC half-hour observation timestamp; supply 48 consecutive rows in ascending order ending 30 minutes before issue.")
    available_at_utc: datetime = Field(description="Latest UTC availability time for this history row; must be no later than the request issue time.")
    eirgrid_ie_wind_generation_mw: float = Field(ge=0, description="EirGrid Ireland observed wind generation, MW.")
    eirgrid_ie_demand_mw: float = Field(ge=0, description="EirGrid Ireland observed electricity demand, MW.")
    entsoe_price_eur_mwh: float = Field(description="ENTSO-E electricity price, EUR/MWh; negative values are allowed.")
    eirgrid_snsp_ratio: float = Field(ge=0, le=1, description="EirGrid system non-synchronous penetration as a 0–1 fraction, not percent.")
    eirgrid_all_island_oversupply_mw: float = Field(ge=0, description="EirGrid all-island oversupply, MW.")
    observed_dispatch_down_mwh: float = Field(ge=0, description="Past observed dispatch-down energy in this completed half-hour, MWh; not the future target.")


class V1RawPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    issue_timestamp_utc: datetime = Field(description="UTC issue timestamp on a :00 or :30 boundary. The current raw values and all 48 history rows must have been available by this time.", examples=["2026-01-31T22:30:00Z"])
    forecast_horizon_minutes: ForecastHorizon = Field(description="Predict the half-hour beginning 30 or 60 minutes after issue.")
    flexible_load_capacity_mw: float = Field(default=100, gt=0, le=10000, description="EV/flexible demand capacity in MW used only for recoverable-surplus output.")
    current_observation: V1RawCurrentObservation = Field(description="Original ENTSO-E and EirGrid source values for the completed issue half-hour. No engineered lags, rolling values or future target.")
    history: list[V1RawHistoryObservation] = Field(min_length=48, max_length=48, description="Exactly 48 consecutive preceding raw half-hour observations (24 hours of prior context), oldest first.")

    _validate_issue_timestamp = field_validator("issue_timestamp_utc")(_require_timezone)


class V2RawHourlyForecast(BaseModel):
    model_config = ConfigDict(extra="forbid")

    region: Literal["west", "south", "east", "north"] = Field(description="Forecast region: west=Galway, south=Cork, east=Dublin, north=Belfast.")
    target_hour_utc: datetime = Field(description="One target hour within the requested UTC day, on an exact hourly boundary, ISO 8601 with timezone.")
    forecast_available_at_utc: datetime = Field(description="When this FORECAST (not realised weather) was published, UTC. It must precede the prediction issue time and target day.")
    wind_speed_100m_kmh: float = Field(ge=0, description="Forecast wind speed 100 m above ground, km/h.")
    shortwave_radiation_wm2: float = Field(ge=0, description="Forecast shortwave solar radiation, W/m².")
    temperature_2m_c: float = Field(description="Forecast air temperature 2 m above ground, °C; negative values are allowed.")


class V2RawPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_date_utc: date = Field(description="Whole UTC calendar day to predict, YYYY-MM-DD. Historical/current days require forecasts published by that day's 00:00 UTC; future days can be at most seven days ahead.")
    hourly_forecasts: list[V2RawHourlyForecast] = Field(min_length=96, max_length=96, description="Exactly 96 source forecast rows: 24 UTC target hours for each of four regions. Do not submit observed weather or curtailment labels.")


class DatasetPredictionRequest(BaseModel):
    issue_timestamp_utc: datetime = Field(
        description=(
            "An available dataset issue time in ISO 8601 format with timezone. "
            "Use GET /dataset/info and GET /dataset/available-times to choose one."
        ),
        examples=["2026-01-15T12:30:00Z"],
    )
    forecast_horizon_minutes: ForecastHorizon = Field(
        description="Supported forecast horizon: 30 or 60 minutes."
    )
    flexible_load_capacity_mw: float = Field(
        default=100.0,
        gt=0,
        le=10000,
        description="Flexible demand capacity used to calculate recoverable surplus.",
    )

    _validate_issue_timestamp = field_validator("issue_timestamp_utc")(_require_timezone)


class DatasetWindowPredictionRequest(BaseModel):
    start_timestamp_utc: datetime = Field(
        description=(
            "First historical issue time in the rolling window. It must be an available "
            "half-hour dataset timestamp in ISO 8601 format with timezone."
        ),
        examples=["2026-01-15T12:00:00Z"],
    )
    duration_hours: float = Field(
        default=2.0,
        ge=0.5,
        le=24,
        multiple_of=0.5,
        description="Window length from 0.5 to 24 hours, in half-hour increments.",
        examples=[2, 24],
    )
    forecast_horizons_minutes: list[ForecastHorizon] = Field(
        default_factory=lambda: [30, 60],
        min_length=1,
        max_length=2,
        description=(
            "Short-horizon predictions to calculate at every half-hour issue time. "
            "Choose 30, 60, or both."
        ),
    )
    flexible_load_capacity_mw: float = Field(
        default=100.0, gt=0, le=10000,
        description="Optional EV/flexible-load capacity in MW used for each v1 surplus estimate.",
    )

    _validate_start_timestamp = field_validator("start_timestamp_utc")(_require_timezone)


class FeaturePredictionRequest(BaseModel):
    issue_timestamp_utc: datetime = Field(
        description="V1 issue time in timezone-aware ISO 8601 UTC format; this is when the complete feature snapshot is known.",
        examples=["2026-01-15T12:30:00Z"],
    )
    forecast_horizon_minutes: ForecastHorizon = Field(description="V1 target is 30 or 60 minutes after the issue time.")
    flexible_load_capacity_mw: float = Field(default=100.0, gt=0, le=10000, description="Optional EV/flexible-load capacity in MW.")
    features: dict[str, float] = Field(
        description="Complete V1 numeric feature snapshot. GET /model-info lists every required_live_features key; a small partial object is not sufficient."
    )

    _validate_issue_timestamp = field_validator("issue_timestamp_utc")(_require_timezone)


class PredictionResponse(BaseModel):
    model_version: str = Field(description="V1 short-horizon model version.")
    issue_timestamp_utc: str = Field(description="UTC time at which the V1 forecast is made.")
    target_timestamp_utc: str = Field(description="UTC half-hour whose dispatch-down outcome is predicted; use this key for /actuals/v1.")
    forecast_horizon_minutes: int = Field(description="Minutes between issue and target time: 30 or 60.")
    dispatch_down_probability: float = Field(description="Estimated probability that any renewable dispatch-down occurs at the target half-hour.")
    dispatch_down_event_prediction: bool = Field(description="Yes/no flag after applying classification_threshold.")
    classification_threshold: float = Field(description="Probability threshold used for the yes/no event flag.")
    risk_level: Literal["low", "medium", "high"] = Field(description="Display category based on predicted dispatch-down risk.")
    predicted_dispatch_down_mwh: float = Field(description="Predicted total dispatched-down renewable energy in the target half-hour, MWh.")
    predicted_curtailment_mwh: float = Field(description="Predicted curtailment component of V1 dispatch-down, MWh; not a whole-day V2 prediction.")
    predicted_constraint_mwh: float = Field(description="Predicted network-constraint component of V1 dispatch-down, MWh.")
    prediction_interval_p10_mwh: float = Field(description="Lower uncertainty estimate for dispatch-down MWh.")
    prediction_interval_p50_mwh: float = Field(description="Middle uncertainty estimate for dispatch-down MWh.")
    prediction_interval_p90_mwh: float = Field(description="Upper uncertainty estimate for dispatch-down MWh.")
    flexible_load_capacity_mw: float = Field(description="Flexible-load capacity supplied in the request, MW.")
    recoverable_surplus_mwh: float = Field(description="Estimated V1 dispatch-down energy that the flexible load could absorb, MWh.")


class LatestPredictionsResponse(BaseModel):
    predictions: list[PredictionResponse]


class DailyCurtailmentRequest(BaseModel):
    target_date_utc: date = Field(
        description="V2 target UTC calendar day in YYYY-MM-DD format, from 2024-04-01 through the current UTC day. No future day or feature object. The forecast is issued at 00:00 UTC on that day.",
        examples=["2026-08-30"],
    )


class ForwardDailyWindowRequest(BaseModel):
    start_date_utc: date = Field(
        description="First full future UTC day to predict, YYYY-MM-DD. Earliest: tomorrow UTC; latest: seven days after today UTC. The last requested day must also be within that range.",
        examples=["2026-09-28"],
    )
    days: int = Field(
        default=7, ge=1, le=7,
        description="Number of consecutive full UTC days, 1–7. Defaults to 7; the whole window must end no later than seven days after today UTC.",
        examples=[7],
    )


class DailyCurtailmentResponse(BaseModel):
    model_version: str = Field(description="V2 daily curtailment model version.")
    experimental: bool = Field(description="True: V2 remains experimental and should not be used as a dispatch instruction.")
    target_date_utc: str = Field(description="UTC day predicted; copy this to /actuals/daily-curtailment for the observed outcome.")
    issue_timestamp_utc: str = Field(description="00:00 UTC at the start of the target day.")
    forecast_max_available_at_utc: str = Field(description="Latest assumed availability time among the archived forecast inputs, before issue time.")
    curtailment_event_probability: float = Field(description="Probability that any curtailment occurs during this UTC day, between 0 and 1.")
    predicted_curtailment_mwh: float = Field(description="Estimated total curtailment energy over the full UTC day, MWh; not an hourly series.")
    target: str = Field(description="Plain-language definition of the V2 prediction target.")


class V1RawPredictionResponse(PredictionResponse):
    input_provenance: Literal["user_supplied_unverified"] = Field(description="Source values and their claimed availability timestamps came from the caller, not a verified live collector.")
    engineered_feature_count: int = Field(description="Number of V1 model features recreated from the supplied raw observations, including horizon.")
    input_notice: str = Field(description="Publication-time and training-distribution limitations of manual V1 inputs.")


class V2RawPredictionResponse(DailyCurtailmentResponse):
    issue_timestamp_utc: str = Field(description="For a future day, actual UTC request time; for a historical/current day, target-day 00:00 UTC using only forecasts claimed available then.")
    forecast_max_available_at_utc: str = Field(description="Latest claimed forecast publication time in the 96 source rows, no later than issue_timestamp_utc.")
    input_provenance: Literal["user_supplied_unverified"] = Field(description="Forecast values/publication times came from the caller; the API validates shape and timestamps but cannot authenticate source provenance.")
    engineered_feature_count: int = Field(description="Number of V2 weather and calendar model features recreated from the 96 hourly inputs.")
    input_notice: str = Field(description="Limitations of user-supplied forecasts and unvalidated multi-day leads.")


class ForwardDailyPredictionResponse(DailyCurtailmentResponse):
    issue_timestamp_utc: str = Field(description="Actual UTC request time after all live forecasts were retrieved; distinct from the target day's midnight.")
    forecast_max_available_at_utc: str = Field(description="Latest actual UTC retrieval timestamp among the four live regional weather forecasts; never after issue_timestamp_utc.")
    forecast_lead_hours: float = Field(description="Hours from request issue time to the target day's 00:00 UTC, not a validated model training lead.")


class ForwardDailyWindowResponse(BaseModel):
    model_version: str = Field(description="Experimental V2 daily curtailment model version, not V1.")
    semantics: Literal["live_forward_daily_curtailment"]
    experimental: Literal[True]
    notice: str = Field(description="Warning that multi-day forecast-lead accuracy has not been validated.")
    forecast_source: str = Field(description="Uncached live Open-Meteo GFS weather forecast; no future observed weather or EirGrid outcomes are used.")
    issued_at_utc: str = Field(description="UTC issue time after all regional live forecasts were retrieved.")
    start_date_utc: str = Field(description="First predicted full UTC calendar day.")
    end_date_utc: str = Field(description="Last predicted full UTC calendar day, inclusive.")
    prediction_count: int = Field(description="Number of daily predictions in the ordered array; equals days.")
    predictions: list[ForwardDailyPredictionResponse] = Field(description="One experimental probability and full-day MWh estimate per requested future UTC day. Use target_date_utc with /actuals/daily-curtailment once observations arrive.")


class HealthResponse(BaseModel):
    status: Literal["ready"] = Field(description="V1 service readiness; v2 can be unavailable while this remains ready.")
    model_version: str = Field(description="Active V1 model version, not the V2 version.")
    dataset_loaded: bool = Field(description="Whether the historical V1 replay dataset is loaded.")
    daily_model_available: bool = Field(description="Whether the optional V2 daily model loaded successfully.")


class DailyCoveragePartition(BaseModel):
    first_target_date_utc: date = Field(description="First complete UTC day in this historical split.")
    last_target_date_utc: date = Field(description="Last complete UTC day in this historical split.")
    complete_day_count: int = Field(description="Number of complete, valid 48-half-hour days in this split.")


class DailyDatasetCoverageResponse(BaseModel):
    model_version: str = Field(description="Experimental daily V2 model, not V1.")
    historical_date_min_utc: date = Field(description="First complete day in the frozen V2 development/evaluation dataset.")
    historical_date_max_utc: date = Field(description="Last complete day in that frozen dataset; not the latest requestable day.")
    historical_complete_day_count: int = Field(description="Total complete labelled days across training, validation and test splits.")
    partitions: dict[str, DailyCoveragePartition] = Field(description="Historical train, validation and untouched test coverage. The fitted model uses train plus validation, not test labels.")
    fitted_through_date_utc: date = Field(description="Last UTC day whose label was used to fit the deployed model.")
    requestable_date_min_utc: date = Field(description="Earliest date supported by the archived forecast-source convention.")
    requestable_date_max_utc: date = Field(description="Current UTC day; future days cannot be requested.")
    forward_window_date_min_utc: date = Field(description="Tomorrow UTC: earliest complete day requestable from the live V2 window route.")
    forward_window_date_max_utc: date = Field(description="Today UTC plus seven days: latest live V2 window target day.")
    maximum_forward_window_days: Literal[7] = Field(description="Maximum number of consecutive full future UTC days in one V2 window request.")
    date_format: Literal["YYYY-MM-DD"]
    timezone: Literal["UTC"]
    dataset_sha256: str = Field(description="Checksum of the frozen historical V2 modelling dataset.")
    forecast_source: str = Field(description="Sources used for V2 labels and weather forecasts.")
    request_notice: str = Field(description="Why historical dataset coverage and requestable-date range differ; actuals have separate coverage.")


class PointActualResponse(BaseModel):
    status: Literal["available", "pending", "missing"] = Field(description="Available: valid observed half-hour; pending: later than archive; missing: no valid row inside coverage.")
    target_timestamp_utc: str = Field(description="V1 target half-hour looked up.")
    source_latest_timestamp_utc: str = Field(description="Latest half-hour timestamp in the bundled actuals snapshot.")
    actual_dispatch_down_mwh: float | None = Field(description="Observed dispatch-down MWh, or null when unavailable.")
    actual_curtailment_mwh: float | None = Field(description="Observed curtailment component MWh, or null when unavailable.")
    actual_constraint_mwh: float | None = Field(description="Observed constraint component MWh, or null when unavailable.")
    actual_dispatch_down_event: bool | None = Field(description="Whether observed dispatch-down was positive, or null when unavailable.")


class DailyActualResponse(BaseModel):
    status: Literal["available", "pending", "missing"] = Field(description="Available only when all 48 UTC half-hours have valid labels; otherwise pending or missing.")
    target_date_utc: str = Field(description="V2 target UTC day looked up.")
    source_latest_timestamp_utc: str = Field(description="Latest half-hour timestamp in the bundled actuals snapshot.")
    actual_curtailment_mwh: float | None = Field(description="Observed full-day curtailment MWh, or null when unavailable.")
    actual_curtailment_event: bool | None = Field(description="Whether observed full-day curtailment was positive, or null when unavailable.")
    complete_half_hour_count: int = Field(description="48 for an available complete UTC day, otherwise 0.")


class ActualsCoverageResponse(BaseModel):
    source: str = Field(description="EirGrid observation archive, separate from either prediction model.")
    available_target_timestamp_min_utc: str = Field(description="Earliest stored half-hour target for V1 actual-value lookups.")
    available_target_timestamp_max_utc: str = Field(description="Latest stored half-hour target for V1 actual-value lookups.")
    complete_day_min_utc: str | None = Field(description="Earliest complete UTC day for V2 actual-value lookups.")
    complete_day_max_utc: str | None = Field(description="Latest complete UTC day for V2 actual-value lookups.")
    notice: str = Field(description="Archive refresh limitations; this endpoint does not describe model-training coverage.")


class PointActualBatchRequest(BaseModel):
    target_timestamps_utc: list[datetime] = Field(
        min_length=1,
        max_length=200,
        description="Copy target_timestamp_utc from 1–200 V1 prediction results, in any order. Each value needs an ISO 8601 timezone and UTC half-hour alignment.",
        examples=[["2026-01-15T12:30:00Z", "2026-01-15T13:00:00Z"]],
    )

    @field_validator("target_timestamps_utc")
    @classmethod
    def validate_times(cls, values: list[datetime]) -> list[datetime]:
        return [_require_timezone(value) for value in values]


class PointActualBatchResponse(BaseModel):
    count: int
    actuals: list[PointActualResponse]


class DatasetInfoResponse(BaseModel):
    available_issue_timestamp_min_utc: str
    available_issue_timestamp_max_utc: str
    available_issue_timestamp_count: int
    available_date_min_utc: str
    available_date_max_utc: str
    timezone: Literal["UTC"]
    timestamp_format: str
    timestamp_example: str
    interval_minutes: int
    required_minute_values: list[int]
    supported_forecast_horizons_minutes: list[int]
    maximum_window_hours: float
    window_semantics: Literal["historical_rolling_short_horizon"]
    window_notice: str


class AvailableTimesResponse(DatasetInfoResponse):
    issue_timestamps_utc: list[str]


class WindowHorizonSummary(BaseModel):
    interval_count: int
    average_dispatch_down_probability: float
    total_predicted_dispatch_down_mwh: float
    peak_predicted_dispatch_down_mwh: float
    total_recoverable_surplus_mwh: float


class DatasetWindowPredictionResponse(BaseModel):
    model_version: str
    semantics: Literal["historical_rolling_short_horizon"]
    notice: str
    start_timestamp_utc: str
    end_timestamp_exclusive_utc: str
    duration_hours: float
    interval_minutes: int
    forecast_horizons_minutes: list[int]
    prediction_count: int
    summary_by_horizon: dict[str, WindowHorizonSummary]
    predictions: list[PredictionResponse]


def _default_service() -> PredictionService:
    return PredictionService(
        model_path=os.getenv("GRIDTOEV_MODEL_PATH", str(DEFAULT_MODEL_PATH)),
        dataset_path=os.getenv("GRIDTOEV_DATASET_PATH", str(DEFAULT_DATASET_PATH)),
        release_report_path=os.getenv("GRIDTOEV_RELEASE_REPORT_PATH", str(DEFAULT_REPORT_PATH)),
    )


def create_app(
    service: PredictionService | None = None,
    api_key: str | None = None,
    *,
    daily_service: DailyCurtailmentService | None = None,
    actuals_service: ActualsService | None = None,
) -> FastAPI:
    prediction_service = service or _default_service()
    daily_model_path = os.getenv("GRIDTOEV_DAILY_MODEL_PATH")
    optional_daily_service = daily_service or (
        DailyCurtailmentService(
            daily_model_path,
            os.getenv("GRIDTOEV_DAILY_REPORT_PATH", str(DEFAULT_REPORT)),
        ) if daily_model_path else None
    )
    observation_service = actuals_service or ActualsService(
        os.getenv("GRIDTOEV_ACTUALS_PATH", str(DEFAULT_HISTORY))
    )
    configured_api_key = (
        api_key if api_key is not None else os.getenv("GRIDTOEV_API_KEY", "")
    ).strip()
    api_key_header = APIKeyHeader(
        name="X-API-Key",
        auto_error=False,
        description="Shared team key for protected V1, V2 and actual-value routes. Use the Authorize button once in Swagger UI.",
    )

    def require_api_key(
        provided_api_key: str | None = Security(api_key_header),
    ) -> None:
        """Require the shared key only when the deployment configured one."""
        if not configured_api_key:
            return
        if provided_api_key is None or not secrets.compare_digest(
            provided_api_key,
            configured_api_key,
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid API key",
                headers={"WWW-Authenticate": "ApiKey"},
            )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not prediction_service.loaded:
            prediction_service.load()
        app.state.prediction_service = prediction_service
        app.state.daily_curtailment_service = None
        if optional_daily_service is not None:
            try:
                optional_daily_service.load()
            except Exception as error:
                logger.warning("Optional daily model unavailable; v1 remains ready: %s", type(error).__name__)
            else:
                app.state.daily_curtailment_service = optional_daily_service
        yield

    app = FastAPI(
        title="GridToEV V1 + V2 Prediction API",
        version="1.1.0",
        description=API_DESCRIPTION,
        openapi_tags=OPENAPI_TAGS,
        lifespan=lifespan,
    )

    configured_origins = os.getenv(
        "GRIDTOEV_CORS_ORIGINS",
        "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173",
    )
    origins = [origin.strip() for origin in configured_origins.split(",") if origin.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, Any]:
        return {
            "service": "GridToEV Prediction API",
            "status": "ready" if prediction_service.loaded else "starting",
            "model_version": (
                prediction_service.metadata["model_version"]
                if prediction_service.loaded
                else None
            ),
            "docs_url": "/docs",
            "health_url": "/health",
            "dataset_info_url": "/dataset/info",
            "api_key_required": bool(configured_api_key),
        }

    @app.get(
        "/health", response_model=HealthResponse, tags=[TAG_SERVICE],
        summary="Check V1 readiness and whether optional V2 loaded",
        description="No input or API key. `status=ready` means V1 is serving; `daily_model_available` independently tells you whether experimental daily V2 loaded.",
    )
    def health() -> dict[str, Any]:
        return {
            "status": "ready",
            "model_version": prediction_service.metadata["model_version"],
            "dataset_loaded": prediction_service.dataset is not None,
            "daily_model_available": app.state.daily_curtailment_service is not None,
        }

    @app.get(
        "/models/catalog", dependencies=[Depends(require_api_key)], tags=[TAG_SERVICE],
        summary="List V1 and V2 targets, availability and detailed model-info routes",
        description="No input. Lists the two different prediction tasks and where to read estimator names, held-out scores and raw-input contracts. Do not compare their MWh MAEs directly: V1 is per half-hour dispatch-down; V2 is per full-day curtailment.",
    )
    def models_catalog() -> dict[str, Any]:
        daily = app.state.daily_curtailment_service
        return {
            "models": [
                {
                    "model_id": "v1", "available": True,
                    "model_version": prediction_service.metadata["model_version"],
                    "target": "Dispatch-down event and MWh per half-hour, 30 or 60 minutes ahead",
                    "details_endpoint": "/model-info",
                    "raw_input_schema_endpoint": "/model-info/v1/raw-input-schema",
                    "raw_prediction_endpoint": "/predict/v1/from-raw",
                },
                {
                    "model_id": "v2", "available": daily is not None,
                    "model_version": daily.bundle["metadata"]["model_version"] if daily is not None else None,
                    "target": "Curtailment event and total MWh over a full UTC day",
                    "details_endpoint": "/model-info/daily-curtailment",
                    "raw_input_schema_endpoint": "/model-info/daily-curtailment/raw-input-schema",
                    "raw_prediction_endpoint": "/predict/curtailment/from-raw",
                },
            ],
            "comparison_notice": "Different targets, energy intervals and test sets: do not compare V1 and V2 MWh MAEs numerically.",
        }

    @app.get(
        "/model-info", dependencies=[Depends(require_api_key)], tags=[TAG_V1],
        summary="V1 model version, required live features and historical dataset range",
        description="No input. Returns V1's seven fitted estimator names and roles, saved held-out and baseline metrics, serving blend weights, caveats and feature contract. The main MWh result is not necessarily the ML regressor's output. For original-source manual input use `/model-info/v1/raw-input-schema` and `/predict/v1/from-raw`.",
    )
    def model_info() -> dict[str, Any]:
        return prediction_service.model_info()

    @app.get(
        "/model-info/v1/raw-input-schema",
        dependencies=[Depends(require_api_key)], tags=[TAG_V1],
        summary="V1 raw source-value field guide, units and 48-row history contract",
        description="No input. Explains every current ENTSO-E/EirGrid source field and the five repeated history signals plus past observed dispatch-down. Use the typed `/predict/v1/from-raw` request body in Swagger to enter values. No engineered lags, rolling columns or future target are requested.",
    )
    def v1_raw_input_schema() -> dict[str, Any]:
        return {
            "model_id": "v1",
            "prediction_target": "Dispatch-down 30 or 60 minutes after the issue half-hour",
            "required_history_half_hours": 48,
            "current_fields": V1RawCurrentObservation.model_json_schema()["properties"],
            "history_fields": V1RawHistoryObservation.model_json_schema()["properties"],
            "input_notice": (
                "Enter original completed-interval source values, not model-engineered features or the future target. "
                "The latest observed dispatch-down is a past input. Source availability times must be no later "
                "than issue time, but caller-provided provenance cannot be authenticated by this endpoint. "
                "The three source-imputation flags are zero because missing raw values are rejected. "
                "A true current-time forecast also requires a current, causally available source feed."
            ),
            "swagger_example_notice": (
                "The prefilled POST /predict/v1/from-raw body uses the latest frozen dataset "
                "observations to predict the first half-hour after the dataset's final labelled "
                "target. It does not add that target or an outcome to the dataset. Publication "
                "times are illustrative and not verified; replace them with genuine source "
                "availability times for a new prediction."
            ),
        }

    @app.get(
        "/actuals/coverage",
        response_model=ActualsCoverageResponse,
        dependencies=[Depends(require_api_key)],
        tags=[TAG_ACTUALS],
        summary="Actuals archive coverage for V1 half-hours and V2 complete days",
        description="No input. This EirGrid observation snapshot is separate from both modelling datasets. Half-hour bounds guide V1 actual lookups; complete-day bounds guide V2 actual lookups. Later actuals are pending until the archive is refreshed.",
    )
    def actuals_coverage() -> dict[str, Any]:
        try:
            return observation_service.coverage()
        except (OSError, ValueError) as error:
            logger.warning("Actuals archive unavailable: %s", type(error).__name__)
            raise HTTPException(status_code=503, detail="Actuals archive unavailable") from error

    @app.get(
        "/actuals/v1",
        response_model=PointActualResponse,
        dependencies=[Depends(require_api_key)],
        tags=[TAG_ACTUALS],
        summary="V1 actual: observed dispatch-down for one target half-hour",
        description="Input: `target_timestamp_utc` from a **V1 prediction response**, not `issue_timestamp_utc`. Use timezone-aware ISO 8601 on a UTC half-hour boundary. `available` returns observed dispatch-down, curtailment and constraint MWh; `pending`/`missing` return null actuals, never a guessed zero.",
    )
    def point_actual(
        target_timestamp_utc: Annotated[
            datetime,
            Query(
                description="V1 target timestamp copied from a prediction; ISO 8601 timezone required, aligned to a UTC half-hour.",
                examples=["2026-01-15T12:30:00Z"],
            ),
        ],
    ) -> dict[str, Any]:
        try:
            return observation_service.point(_require_timezone(target_timestamp_utc))
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except OSError as error:
            logger.warning("Actuals archive unavailable: %s", type(error).__name__)
            raise HTTPException(status_code=503, detail="Actuals archive unavailable") from error

    @app.post(
        "/actuals/v1/batch",
        response_model=PointActualBatchResponse,
        dependencies=[Depends(require_api_key)],
        tags=[TAG_ACTUALS],
        summary="V1 actuals: look up 1–200 prediction target half-hours",
        description="Input: `target_timestamps_utc` array copied from V1 prediction results, in ISO 8601 UTC format. Returns observed actuals in request order. `pending` and `missing` are not interpreted as zero.",
    )
    def point_actual_batch(request: PointActualBatchRequest) -> dict[str, Any]:
        try:
            rows = observation_service.points(request.target_timestamps_utc)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except OSError as error:
            logger.warning("Actuals archive unavailable: %s", type(error).__name__)
            raise HTTPException(status_code=503, detail="Actuals archive unavailable") from error
        return {"count": len(rows), "actuals": rows}

    @app.get(
        "/actuals/daily-curtailment",
        response_model=DailyActualResponse,
        dependencies=[Depends(require_api_key)],
        tags=[TAG_ACTUALS],
        summary="V2 actual: observed curtailment for one complete UTC day",
        description="Input: `target_date_utc` copied from a **V2 daily prediction**, formatted `YYYY-MM-DD`. An actual is `available` only after all 48 EirGrid half-hour labels for that UTC day are valid. Otherwise `pending` or `missing` returns null actuals.",
    )
    def daily_actual(
        target_date_utc: Annotated[
            date,
            Query(description="V2 target UTC date copied from a daily prediction; YYYY-MM-DD.", examples=["2026-08-30"]),
        ],
    ) -> dict[str, Any]:
        try:
            return observation_service.day(target_date_utc)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except OSError as error:
            logger.warning("Actuals archive unavailable: %s", type(error).__name__)
            raise HTTPException(status_code=503, detail="Actuals archive unavailable") from error

    @app.get(
        "/model-info/daily-curtailment", dependencies=[Depends(require_api_key)], tags=[TAG_V2],
        summary="V2 model version, weather features and historical test metrics",
        description="No input. Returns V2 classifier/regressor names, candidate selection, daily held-out event/amount scores, baselines, date splits and the unvalidated multi-day lead warning. V1 half-hour dispatch-down MAE is not comparable. Returns 503 when V2 is not loaded.",
    )
    def daily_model_info() -> dict[str, Any]:
        available_service = app.state.daily_curtailment_service
        if available_service is None:
            raise HTTPException(status_code=503, detail="Optional daily model is unavailable")
        return available_service.model_info()

    @app.get(
        "/model-info/daily-curtailment/raw-input-schema",
        dependencies=[Depends(require_api_key)], tags=[TAG_V2],
        summary="V2 original hourly weather forecast input guide and required regions",
        description="No input. Explains the exact 96 source forecast rows needed for one daily raw-input prediction: 24 UTC hours each for Galway/west, Cork/south, Dublin/east and Belfast/north. Enter forecasts, not realised weather or curtailment labels.",
    )
    def v2_raw_input_schema() -> dict[str, Any]:
        if app.state.daily_curtailment_service is None:
            raise HTTPException(status_code=503, detail="Optional daily model is unavailable")
        return {
            "model_id": "v2",
            "required_hourly_rows": 96,
            "required_regions": {
                "west": "Galway", "south": "Cork", "east": "Dublin", "north": "Belfast",
            },
            "hours_per_region": 24,
            "hourly_fields": V2RawHourlyForecast.model_json_schema()["properties"],
            "input_notice": (
                "All 96 rows must be weather forecasts for the chosen UTC day, with publication times "
                "no later than prediction issue. The API checks completeness but cannot authenticate "
                "user-supplied forecast vintages. For future days, longer-lead accuracy is unvalidated."
            ),
            "swagger_example_notice": (
                "The prefilled POST /predict/curtailment/from-raw body uses archived "
                "Open-Meteo GFS previous_day1 forecasts for the first day after the frozen "
                "V2 model dataset (2026-08-31). It does not add that day or its outcome to "
                "the dataset. For a different day, replace all 96 regional hours with "
                "genuine forecasts and their actual publication times."
            ),
        }

    @app.get(
        "/dataset/daily-curtailment/coverage",
        response_model=DailyDatasetCoverageResponse,
        dependencies=[Depends(require_api_key)],
        tags=[TAG_V2],
        summary="V2 daily dataset dates, split counts and requestable date range",
        description="No input. Shows the fixed historical train/validation/test coverage used to build and evaluate V2, the archived `/predict/curtailment/day` request range, and tomorrow-through-seven-days-ahead limits for the live `/predict/curtailment/window` route. Future dates are **not** additional trained rows and source weather may be missing. For observed-outcome coverage use `/actuals/coverage`. Returns 503 when V2 is not loaded.",
    )
    def daily_dataset_coverage() -> dict[str, Any]:
        available_service = app.state.daily_curtailment_service
        if available_service is None:
            raise HTTPException(status_code=503, detail="Optional daily model is unavailable")
        return available_service.dataset_coverage()

    @app.post(
        "/predict/curtailment/day",
        response_model=DailyCurtailmentResponse,
        dependencies=[Depends(require_api_key)],
        tags=[TAG_V2],
        summary="V2: predict curtailment probability and total MWh for one UTC day",
        description='Body: `{"target_date_utc":"YYYY-MM-DD"}` for 2024-04-01 through **today UTC**; no weather fields or forecast horizon are supplied. V2 builds archived day-ahead weather features and predicts an event probability plus total curtailment MWh for the whole day, **not hourly points**. A future day or incomplete forecast yields 422; unavailable forecast service or V2 model yields 503. Copy `target_date_utc` to `/actuals/daily-curtailment` when checking the outcome.',
    )
    def predict_daily_curtailment(request: DailyCurtailmentRequest) -> dict[str, Any]:
        available_service = app.state.daily_curtailment_service
        if available_service is None:
            raise HTTPException(status_code=503, detail="Optional daily model is unavailable")
        try:
            return available_service.predict_date(request.target_date_utc)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except (OSError, TimeoutError) as error:
            raise HTTPException(status_code=503, detail=f"Forecast source unavailable: {error}") from error

    @app.post(
        "/predict/curtailment/from-raw",
        response_model=V2RawPredictionResponse,
        dependencies=[Depends(require_api_key)], tags=[TAG_V2],
        summary="V2: predict one full UTC day from 96 original hourly weather forecasts",
        description=(
            "Body: `target_date_utc` and exactly 24 hourly FORECAST records for each of west/Galway, "
            "south/Cork, east/Dublin and north/Belfast. Each record needs target hour, publication time, "
            "100 m wind speed in km/h, shortwave radiation in W/m² and 2 m temperature in °C. "
            "No engineered aggregates or curtailment target. Historical/current days require vintage times "
            "by that day's 00:00 UTC; future days may be at most seven days ahead and use request time. "
            "The response is one whole-day V2 prediction, NOT an hourly series. Caller-supplied provenance "
            "is unverified, and historical V2 MAE does not validate longer future leads. See "
            "`/model-info/daily-curtailment/raw-input-schema` for region and field guidance. "
            "The Swagger body demonstrates the next unrecorded V2 model-dataset day, "
            "2026-08-31, using its archived day-ahead forecasts. This is not a forecast "
            "for today's clock time and no outcome is added to the dataset. To predict "
            "another day, change the date and replace all 96 forecast rows."
        ),
    )
    def predict_daily_curtailment_from_raw(request: V2RawPredictionRequest) -> dict[str, Any]:
        available_service = app.state.daily_curtailment_service
        if available_service is None:
            raise HTTPException(status_code=503, detail="Optional daily model is unavailable")
        now = datetime.now(timezone.utc)
        target_date = request.target_date_utc
        if target_date < EARLIEST_TARGET_DATE or target_date > now.date() + timedelta(days=7):
            raise HTTPException(status_code=422, detail="target_date_utc must be from 2024-04-01 through seven days after today UTC")
        issued_at = (
            now if target_date > now.date()
            else datetime.combine(target_date, datetime.min.time(), tzinfo=timezone.utc)
        )
        try:
            features = build_v2_features_from_raw(
                target_date, [row.model_dump() for row in request.hourly_forecasts], issued_at,
            )
            required = set(available_service.bundle["metadata"]["feature_columns"])
            available = set(features) - {"issue_timestamp_utc", "forecast_max_available_at_utc"}
            if available != required:
                raise HTTPException(status_code=503, detail="Raw V2 adapter does not match the loaded model feature contract")
            result = available_service.predict_features(features, issued_at_utc=issued_at)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return {
            **result,
            "input_provenance": "user_supplied_unverified",
            "engineered_feature_count": len(required),
            "input_notice": (
                "User-supplied forecast values and publication times are not authenticated. "
                "Historical evaluation assumes a specific 24-hour-lead forecast convention; other "
                "vintages and future multi-day leads do not inherit that measured accuracy."
            ),
        }

    @app.post(
        "/predict/curtailment/window",
        response_model=ForwardDailyWindowResponse,
        dependencies=[Depends(require_api_key)],
        tags=[TAG_V2],
        summary="V2: predict curtailment for 1–7 complete future UTC days",
        description=(
            'Body: `{"start_date_utc":"YYYY-MM-DD","days":7}`. Start no earlier than **tomorrow UTC**; '
            "the final day must be no later than **today UTC + 7 days**. Returns an ordered array "
            "with one V2 curtailment-event probability and total-MWh estimate per complete UTC day. "
            "This uses uncached live GFS weather from the request time, not the historical V2 dataset, "
            "future observations or V1's 30/60-minute model. The trained V2 model saw fixed 24-hour-lead "
            "weather; accuracy for longer forecast leads has **not been validated**. A bad date or length "
            "returns 422; unavailable or incomplete live forecasts, or an unloaded V2 model, return 503. "
            "Check each `target_date_utc` later at `/actuals/daily-curtailment`."
        ),
    )
    def predict_daily_curtailment_window(request: ForwardDailyWindowRequest) -> dict[str, Any]:
        available_service = app.state.daily_curtailment_service
        if available_service is None:
            raise HTTPException(status_code=503, detail="Optional daily model is unavailable")
        try:
            return available_service.predict_forward_window(request.start_date_utc, request.days)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except (OSError, TimeoutError) as error:
            logger.warning("Live daily forecast unavailable: %s", type(error).__name__)
            raise HTTPException(status_code=503, detail="Live weather forecast unavailable or incomplete") from error

    @app.get(
        "/dataset/info",
        response_model=DatasetInfoResponse,
        dependencies=[Depends(require_api_key)],
        tags=[TAG_V1],
        summary="V1 historical dataset coverage and valid UTC issue-time format",
        description="No input. Returns the loaded **V1** half-hour issue-time range, count, timestamp format, supported 30/60-minute horizons and 24-hour replay limit. These are historical rows, not current live forecasts. For daily V2 coverage use `/dataset/daily-curtailment/coverage`.",
    )
    def dataset_info() -> dict[str, Any]:
        return prediction_service.dataset_info()

    @app.get(
        "/dataset/available-times",
        response_model=AvailableTimesResponse,
        dependencies=[Depends(require_api_key)],
        tags=[TAG_V1],
        summary="V1: list valid historical issue timestamps",
        description="Optional `limit` query (1–1000, default 96) selects how many of the latest **V1 dataset** issue times to return. Copy one to `issue_timestamp_utc` for `/predict/from-dataset`, or use it as a window start. This list is historical, not a live clock.",
    )
    def available_times(
        limit: Annotated[int, Query(ge=1, le=1000, description="Number of latest V1 historical half-hour issue timestamps to list (1–1000).", examples=[48])] = 96,
    ) -> dict[str, Any]:
        return {
            **prediction_service.dataset_info(),
            "issue_timestamps_utc": prediction_service.available_times(limit),
        }

    @app.post(
        "/predict/from-dataset",
        response_model=PredictionResponse,
        dependencies=[Depends(require_api_key)],
        tags=[TAG_V1],
        summary="V1: predict one historical 30- or 60-minute target",
        description="Body: a V1 `issue_timestamp_utc` from `/dataset/info` or `/dataset/available-times`, `forecast_horizon_minutes` of 30 or 60, and optional `flexible_load_capacity_mw`. Returns dispatch-down risk and MWh for the target half-hour. A date alone, an unavailable time or a time outside the loaded historical dataset cannot be used; a missing row returns 404.",
    )
    def predict_from_dataset(request: DatasetPredictionRequest) -> dict[str, Any]:
        try:
            return prediction_service.predict_from_dataset(
                request.issue_timestamp_utc,
                request.forecast_horizon_minutes,
                request.flexible_load_capacity_mw,
            )
        except DatasetSelectionError as error:
            raise HTTPException(status_code=404, detail=error.detail) from error
        except FeatureValidationError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post(
        "/predict/v1/from-raw",
        response_model=V1RawPredictionResponse,
        dependencies=[Depends(require_api_key)], tags=[TAG_V1],
        summary="V1: predict the next 30/60-minute target from original source observations",
        description=(
            "Body: a UTC half-hour `issue_timestamp_utc`, 30/60-minute horizon, a complete current "
            "ENTSO-E/EirGrid source snapshot and exactly 48 consecutive prior half-hour history rows. "
            "Every source value, including the latest PAST observed dispatch-down, must have been available "
            "by issue time. Do not supply future target values or engineered lags/rolling columns. "
            "The API deterministically creates the V1 model's 119 features and predicts the target "
            "half-hour. This is not a live collector and cannot verify the caller's publication timestamps. "
            "See `/model-info/v1/raw-input-schema` for plain-language fields and units. "
            "The Swagger body uses the latest frozen dataset input row at 2026-01-31 "
            "22:30 UTC to predict the next unrecorded target at 23:30 UTC with the "
            "60-minute horizon. No outcome is added to the dataset. This is not a "
            "live current-time forecast; example publication times are illustrative "
            "and unverified. For a new issue time, replace the current values and "
            "every history row with real values known then. Do not backdate availability."
        ),
    )
    def predict_v1_from_raw(request: V1RawPredictionRequest) -> dict[str, Any]:
        try:
            features = build_v1_features_from_raw(
                request.issue_timestamp_utc,
                request.current_observation.model_dump(),
                [row.model_dump() for row in request.history],
                request.forecast_horizon_minutes,
            )
            required = set(prediction_service.bundle["feature_columns"])
            if set(features) != required:
                raise HTTPException(status_code=503, detail="Raw V1 adapter does not match the loaded model feature contract")
            result = prediction_service.predict_features(
                features={key: value for key, value in features.items() if key != "forecast_horizon_minutes"},
                forecast_horizon_minutes=request.forecast_horizon_minutes,
                flexible_load_capacity_mw=request.flexible_load_capacity_mw,
                issue_timestamp_utc=request.issue_timestamp_utc,
            )
        except (ValueError, FeatureValidationError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return {
            **result,
            "input_provenance": "user_supplied_unverified",
            "engineered_feature_count": len(features),
            "input_notice": (
                "All source values and publication times were supplied by the caller. "
                "The API cannot verify that they were truly known at issue time or match training sources."
            ),
        }

    @app.post(
        "/predict/window/from-dataset",
        response_model=DatasetWindowPredictionResponse,
        dependencies=[Depends(require_api_key)],
        tags=[TAG_V1],
        summary="V1: replay 30/60-minute predictions across a historical window",
        description=(
            "Body: a valid historical `start_timestamp_utc`, `duration_hours` from 0.5 to 24 "
            "in 0.5-hour steps, optional 30/60-minute horizons and capacity. Returns an ordered "
            "array of V1 predictions at successive dataset half-hours, with a summary per horizon. "
            "It is **not a day-ahead** or single 2-hour forecast; every array item is still 30 or 60 minutes ahead of its own issue time."
        ),
    )
    def predict_window_from_dataset(
        request: DatasetWindowPredictionRequest,
    ) -> dict[str, Any]:
        try:
            return prediction_service.predict_window_from_dataset(
                start_timestamp_utc=request.start_timestamp_utc,
                duration_hours=request.duration_hours,
                forecast_horizons_minutes=request.forecast_horizons_minutes,
                flexible_load_capacity_mw=request.flexible_load_capacity_mw,
            )
        except DatasetSelectionError as error:
            raise HTTPException(status_code=404, detail=error.detail) from error
        except FeatureValidationError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post(
        "/predict/features",
        response_model=PredictionResponse,
        dependencies=[Depends(require_api_key)],
        tags=[TAG_V1],
        summary="V1: predict from a complete externally supplied feature snapshot",
        description="Advanced V1 endpoint. Body needs a timezone-aware `issue_timestamp_utc`, a 30/60-minute horizon and a `features` object containing **every** numeric key in `/model-info` → `feature_contract.required_live_features`. Unlike dataset replay, this route does not fetch or fill missing features. Invalid or incomplete snapshots return 422; only use values known at the issue time.",
    )
    def predict_features(request: FeaturePredictionRequest) -> dict[str, Any]:
        try:
            return prediction_service.predict_features(
                features=request.features,
                forecast_horizon_minutes=request.forecast_horizon_minutes,
                flexible_load_capacity_mw=request.flexible_load_capacity_mw,
                issue_timestamp_utc=request.issue_timestamp_utc,
            )
        except FeatureValidationError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get(
        "/predict/latest",
        response_model=LatestPredictionsResponse,
        dependencies=[Depends(require_api_key)],
        tags=[TAG_V1],
        summary="V1: predict both horizons for the latest historical dataset row",
        description="Optional `flexible_load_capacity_mw` query (default 100). Returns two V1 predictions, 30 and 60 minutes ahead of the **latest loaded historical issue time**. This is **not a current-time/live forecast**; check `/dataset/info` for its actual timestamp.",
    )
    def predict_latest(
        flexible_load_capacity_mw: Annotated[float, Query(gt=0, le=10000, description="Optional EV/flexible-load capacity in MW for V1 surplus calculations.")] = 100.0,
    ) -> dict[str, Any]:
        try:
            predictions = prediction_service.predict_latest(flexible_load_capacity_mw)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return {"predictions": predictions}

    standard_openapi = app.openapi

    def openapi_with_raw_examples() -> dict[str, Any]:
        """Make Swagger Try-it-out bodies complete rather than fabricated placeholders."""

        schema = standard_openapi()
        v1_example = v1_raw_example(prediction_service.dataset, prediction_service.dataset_path)
        if v1_example is not None:
            schema["paths"]["/predict/v1/from-raw"]["post"]["requestBody"]["content"]["application/json"]["example"] = v1_example
        daily_report_path = optional_daily_service.report_path if optional_daily_service is not None else DEFAULT_REPORT
        v2_example = v2_raw_example(daily_report_path)
        if v2_example is not None:
            schema["paths"]["/predict/curtailment/from-raw"]["post"]["requestBody"]["content"]["application/json"]["example"] = v2_example
        return schema

    app.openapi = openapi_with_raw_examples
    return app


app = create_app()
