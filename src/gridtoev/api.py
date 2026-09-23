from __future__ import annotations

import os
import secrets
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from .constants import DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH
from .inference import FeatureValidationError, PredictionService


ForecastHorizon = Literal[30, 60]


class DatasetPredictionRequest(BaseModel):
    issue_timestamp_utc: datetime
    forecast_horizon_minutes: ForecastHorizon
    flexible_load_capacity_mw: float = Field(default=100.0, gt=0, le=10000)


class FeaturePredictionRequest(BaseModel):
    issue_timestamp_utc: datetime
    forecast_horizon_minutes: ForecastHorizon
    flexible_load_capacity_mw: float = Field(default=100.0, gt=0, le=10000)
    features: dict[str, float]


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


def _default_service() -> PredictionService:
    return PredictionService(
        model_path=os.getenv("GRIDTOEV_MODEL_PATH", str(DEFAULT_MODEL_PATH)),
        dataset_path=os.getenv("GRIDTOEV_DATASET_PATH", str(DEFAULT_DATASET_PATH)),
    )


def create_app(
    service: PredictionService | None = None,
    api_key: str | None = None,
) -> FastAPI:
    prediction_service = service or _default_service()
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
        yield

    app = FastAPI(
        title="GridToEV Prediction API",
        version="1.0.0",
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

    @app.get("/dataset/available-times", dependencies=[Depends(require_api_key)])
    def available_times(
        limit: Annotated[int, Query(ge=1, le=1000)] = 96,
    ) -> dict[str, Any]:
        return {"issue_timestamps_utc": prediction_service.available_times(limit)}

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
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
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
