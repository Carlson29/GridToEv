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
        le=48,
        multiple_of=0.5,
        description="Window length from 0.5 to 48 hours, in half-hour increments.",
        examples=[2, 24, 48],
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
    flexible_load_capacity_mw: float = Field(default=100.0, gt=0, le=10000)

    _validate_start_timestamp = field_validator("start_timestamp_utc")(_require_timezone)


class FeaturePredictionRequest(BaseModel):
    issue_timestamp_utc: datetime = Field(
        description="Feature snapshot time in ISO 8601 format with timezone.",
        examples=["2026-01-15T12:30:00Z"],
    )
    forecast_horizon_minutes: ForecastHorizon
    flexible_load_capacity_mw: float = Field(default=100.0, gt=0, le=10000)
    features: dict[str, float]

    _validate_issue_timestamp = field_validator("issue_timestamp_utc")(_require_timezone)


class PredictionResponse(BaseModel):
    model_version: str
    issue_timestamp_utc: str
    target_timestamp_utc: str
    forecast_horizon_minutes: int
    dispatch_down_probability: float
    dispatch_down_event_prediction: bool
    classification_threshold: float
    risk_level: Literal["low", "medium", "high"]
    predicted_dispatch_down_mwh: float
    predicted_curtailment_mwh: float
    predicted_constraint_mwh: float
    prediction_interval_p10_mwh: float
    prediction_interval_p50_mwh: float
    prediction_interval_p90_mwh: float
    flexible_load_capacity_mw: float
    recoverable_surplus_mwh: float


class LatestPredictionsResponse(BaseModel):
    predictions: list[PredictionResponse]


class DailyCurtailmentRequest(BaseModel):
    target_date_utc: date = Field(
        description="UTC calendar day in YYYY-MM-DD format, from 2024-04-01 through the current UTC day. The forecast is issued at 00:00 UTC on that day."
    )


class DailyCurtailmentResponse(BaseModel):
    model_version: str
    experimental: bool
    target_date_utc: str
    issue_timestamp_utc: str
    forecast_max_available_at_utc: str
    curtailment_event_probability: float
    predicted_curtailment_mwh: float
    target: str


class PointActualResponse(BaseModel):
    status: Literal["available", "pending", "missing"]
    target_timestamp_utc: str
    source_latest_timestamp_utc: str
    actual_dispatch_down_mwh: float | None
    actual_curtailment_mwh: float | None
    actual_constraint_mwh: float | None
    actual_dispatch_down_event: bool | None


class DailyActualResponse(BaseModel):
    status: Literal["available", "pending", "missing"]
    target_date_utc: str
    source_latest_timestamp_utc: str
    actual_curtailment_mwh: float | None
    actual_curtailment_event: bool | None
    complete_half_hour_count: int


class ActualsCoverageResponse(BaseModel):
    source: str
    available_target_timestamp_min_utc: str
    available_target_timestamp_max_utc: str
    complete_day_min_utc: str | None
    complete_day_max_utc: str | None
    notice: str


class PointActualBatchRequest(BaseModel):
    target_timestamps_utc: list[datetime] = Field(
        min_length=1,
        max_length=200,
        description="Copy target_timestamp_utc from up to 200 v1 prediction results, in any order.",
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
    api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

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
        title="GridToEV Prediction API",
        version="1.1.0",
        description=(
            "Forecast Irish renewable dispatch-down risk and energy 30 or 60 minutes ahead."
        ),
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

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ready",
            "model_version": prediction_service.metadata["model_version"],
            "dataset_loaded": prediction_service.dataset is not None,
        }

    @app.get("/model-info", dependencies=[Depends(require_api_key)])
    def model_info() -> dict[str, Any]:
        return prediction_service.model_info()

    @app.get(
        "/actuals/coverage",
        response_model=ActualsCoverageResponse,
        dependencies=[Depends(require_api_key)],
        summary="Show the observation archive's actual-value coverage",
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
        summary="Retrieve an observed v1 half-hour target, or its availability status",
    )
    def point_actual(
        target_timestamp_utc: Annotated[
            datetime,
            Query(description="Copy target_timestamp_utc from a v1 prediction; ISO 8601 timezone required."),
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
        summary="Retrieve actuals for an array of v1 prediction targets",
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
        summary="Retrieve observed daily curtailment, or its availability status",
    )
    def daily_actual(
        target_date_utc: Annotated[
            date,
            Query(description="Copy target_date_utc from a daily-v2 prediction; YYYY-MM-DD."),
        ],
    ) -> dict[str, Any]:
        try:
            return observation_service.day(target_date_utc)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except OSError as error:
            logger.warning("Actuals archive unavailable: %s", type(error).__name__)
            raise HTTPException(status_code=503, detail="Actuals archive unavailable") from error

    @app.get("/model-info/daily-curtailment", dependencies=[Depends(require_api_key)])
    def daily_model_info() -> dict[str, Any]:
        available_service = app.state.daily_curtailment_service
        if available_service is None:
            raise HTTPException(status_code=503, detail="Optional daily model is unavailable")
        return available_service.bundle["metadata"]

    @app.post("/predict/curtailment/day", response_model=DailyCurtailmentResponse, dependencies=[Depends(require_api_key)])
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

    @app.get(
        "/dataset/info",
        response_model=DatasetInfoResponse,
        dependencies=[Depends(require_api_key)],
        summary="Explain the valid dataset dates and timestamp format",
    )
    def dataset_info() -> dict[str, Any]:
        return prediction_service.dataset_info()

    @app.get(
        "/dataset/available-times",
        response_model=AvailableTimesResponse,
        dependencies=[Depends(require_api_key)],
        summary="List valid issue timestamps and dataset date guidance",
    )
    def available_times(
        limit: Annotated[int, Query(ge=1, le=1000)] = 96,
    ) -> dict[str, Any]:
        return {
            **prediction_service.dataset_info(),
            "issue_timestamps_utc": prediction_service.available_times(limit),
        }

    @app.post(
        "/predict/from-dataset",
        response_model=PredictionResponse,
        dependencies=[Depends(require_api_key)],
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
        summary="Return an array of rolling historical short-horizon predictions",
        description=(
            "Starting at a valid dataset timestamp, calculate 30- and/or 60-minute "
            "predictions every half-hour for up to 48 hours. This is a historical rolling "
            "replay, not a single multi-hour or day-ahead forecast."
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
    )
    def predict_latest(
        flexible_load_capacity_mw: Annotated[float, Query(gt=0, le=10000)] = 100.0,
    ) -> dict[str, Any]:
        try:
            predictions = prediction_service.predict_latest(flexible_load_capacity_mw)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return {"predictions": predictions}

    return app


app = create_app()
