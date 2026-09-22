"""FastAPI application exposing the fixed EngineGuard endpoints."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, status

from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    EngineHistoryRequest,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    ReadyResponse,
)
from src.api.service import ModelService, risk_level

logger = logging.getLogger("engineguard.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.model_service = ModelService()
        app.state.startup_error = None
    except Exception as error:
        logger.exception("model_startup_failed")
        app.state.model_service = None
        app.state.startup_error = str(error)
    yield


app = FastAPI(
    title="EngineGuard API",
    version="1.0.0",
    lifespan=lifespan,
)


def _service(request: Request) -> ModelService:
    service = getattr(request.app.state, "model_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service is not ready",
        )
    return service


def _predict_one(
    request: Request,
    history: EngineHistoryRequest,
    request_id: str,
) -> PredictionResponse:
    service = _service(request)
    started = time.perf_counter()
    predicted_rul = service.predict(
        history.unit_id,
        [observation.model_dump() for observation in history.observations],
    )
    response = PredictionResponse(
        unit_id=history.unit_id,
        predicted_rul=predicted_rul,
        risk_level=risk_level(predicted_rul),
        model_version=service.model_version,
        feature_schema_version=service.feature_schema_version,
        request_id=request_id,
        predicted_at=datetime.now(UTC),
    )
    logger.info(
        "prediction_completed",
        extra={
            "request_id": request_id,
            "unit_id": history.unit_id,
            "model_version": service.model_version,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "validation_status": "valid",
        },
    )
    return response


@app.post("/v1/predict", response_model=PredictionResponse)
def predict(history: EngineHistoryRequest, request: Request) -> PredictionResponse:
    return _predict_one(request, history, str(uuid4()))


@app.post("/v1/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(
    batch: BatchPredictionRequest,
    request: Request,
) -> BatchPredictionResponse:
    request_id = str(uuid4())
    predictions = [_predict_one(request, history, request_id) for history in batch.histories]
    return BatchPredictionResponse(predictions=predictions, request_id=request_id)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="healthy")


@app.get("/ready", response_model=ReadyResponse)
def ready(request: Request) -> ReadyResponse:
    service = _service(request)
    return ReadyResponse(
        status="ready",
        model_version=service.model_version,
        feature_schema_version=service.feature_schema_version,
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info(request: Request) -> ModelInfoResponse:
    service = _service(request)
    return ModelInfoResponse(
        model_version=service.model_version,
        feature_schema_version=service.feature_schema_version,
        model_type=type(service.model).__name__,
        target="target_rul",
    )
