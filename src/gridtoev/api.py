from __future__ import annotations

import os
import secrets
import logging
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, field_validator

from .actuals import ActualsService
from .constants import DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH
from .daily_curtailment import DEFAULT_HISTORY
from .daily_model import DEFAULT_REPORT, DailyCurtailmentService
from .inference import DatasetSelectionError, FeatureValidationError, PredictionService
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
| **V1: 30/60-minute** | Renewable **dispatch-down** risk and MWh at one target half-hour, 30 or 60 minutes after an issue time. | A historical dataset issue timestamp and horizon, or a complete live feature snapshot. | `GET /dataset/info` for valid historical dates; `POST /predict/from-dataset` for a simple example. |
| **V2: daily curtailment (experimental)** | Whether **curtailment** occurs and total curtailment MWh over a full UTC calendar day. This is a different target and time scale from v1. | One historical/current UTC date, or a future start date and 1–7 complete days. | `POST /predict/curtailment/day` for archived inputs; `POST /predict/curtailment/window` for live future weather. |

V1's dataset replay endpoints use a **fixed historical dataset**. In particular, `/predict/latest` means the latest *dataset row*, **not the current time**. A V1 window is repeated 30/60-minute predictions for at most 24 historical hours, **not a day-ahead forecast**. V2's `/day` route fetches archived 24-hour-lead weather for dates through today UTC. Its `/window` route fetches **live GFS forecasts** for tomorrow through seven days ahead and returns one prediction per full UTC day. Multi-day lead accuracy has **not been validated** against the historical V2 test metric. The version shown in the Swagger header does not choose a model; each prediction reports its own `model_version`.

Send the shared `X-API-Key` using **Authorize** for protected routes. `GET /health` is public and shows whether optional v2 loaded. To compare a prediction with reality, copy `target_timestamp_utc` from v1 to `/actuals/v1`, or `target_date_utc` from v2 to `/actuals/daily-curtailment`. Actuals come from a separately refreshed EirGrid snapshot and may be `pending` or `missing`.
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
        "/model-info", dependencies=[Depends(require_api_key)], tags=[TAG_V1],
        summary="V1 model version, required live features and historical dataset range",
        description="No query or body input. Returns metadata for the 30/60-minute dispatch-down model, including its full `required_live_features` list for `/predict/features`. For exact replay dates use `/dataset/info`.",
    )
    def model_info() -> dict[str, Any]:
        return prediction_service.model_info()

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
        description="No input. Returns experimental daily-curtailment model metadata and its **daily** test MAE; do not compare this MAE numerically with V1 half-hour dispatch-down MAE. Returns 503 when V2 is not loaded.",
    )
    def daily_model_info() -> dict[str, Any]:
        available_service = app.state.daily_curtailment_service
        if available_service is None:
            raise HTTPException(status_code=503, detail="Optional daily model is unavailable")
        return available_service.bundle["metadata"]

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

    return app


app = create_app()
