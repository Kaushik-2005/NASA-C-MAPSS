"""FastAPI application exposing the fixed EngineGuard endpoints."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import cast
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    EngineHistoryRequest,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    ReadyResponse,
)
from src.api.security import (
    API_KEY_ENV_VAR,
    MAX_REQUEST_BYTES,
    REQUEST_TIMEOUT_SECONDS,
    api_key_is_valid,
    validation_details,
)
from src.api.service import ModelService, risk_level

logger = logging.getLogger("engineguard.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
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


@app.middleware("http")
async def request_safety_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Reject oversized requests and bound request processing time."""
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            request_bytes = int(content_length)
        except ValueError:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "error_code": "INVALID_CONTENT_LENGTH",
                    "message": "Content-Length must be an integer",
                    "request_id": str(uuid4()),
                },
            )
        if request_bytes > MAX_REQUEST_BYTES:
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={
                    "error_code": "REQUEST_TOO_LARGE",
                    "message": f"Request body must be at most {MAX_REQUEST_BYTES} bytes",
                    "request_id": str(uuid4()),
                },
            )

    if request.url.path.startswith("/v1/") and not api_key_is_valid(
        request.headers.get("x-api-key")
    ):
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "error_code": "AUTHENTICATION_REQUIRED",
                "message": f"Provide a valid API key in X-API-Key when {API_KEY_ENV_VAR} is configured",
                "request_id": str(uuid4()),
            },
        )

    try:
        return await asyncio.wait_for(
            call_next(request),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        return JSONResponse(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            content={
                "error_code": "REQUEST_TIMEOUT",
                "message": "Request processing exceeded the service timeout",
                "request_id": str(uuid4()),
            },
        )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Return a stable validation error without echoing sensor histories."""
    request_id = request.headers.get("x-request-id", str(uuid4()))
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error_code": "VALIDATION_ERROR",
            "message": "Request failed schema validation",
            "request_id": request_id,
            "details": validation_details(exc.errors()),
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Return typed errors for service and readiness failures."""
    request_id = request.headers.get("x-request-id", str(uuid4()))
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error_code": "HTTP_ERROR",
            "message": str(exc.detail),
            "request_id": request_id,
        },
        headers=exc.headers,
    )


def _service(request: Request) -> ModelService:
    service = getattr(request.app.state, "model_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service is not ready",
        )
    return cast(ModelService, service)


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
