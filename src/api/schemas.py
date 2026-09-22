"""Typed request and response contracts for the EngineGuard API."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    field_validator,
    model_validator,
)


class Observation(BaseModel):
    """One ordered engine observation."""

    model_config = ConfigDict(extra="forbid")

    cycle: StrictInt = Field(gt=0)
    op_setting_1: StrictFloat
    op_setting_2: StrictFloat
    op_setting_3: StrictFloat
    sensor_1: StrictFloat
    sensor_2: StrictFloat
    sensor_3: StrictFloat
    sensor_4: StrictFloat
    sensor_5: StrictFloat
    sensor_6: StrictFloat
    sensor_7: StrictFloat
    sensor_8: StrictFloat
    sensor_9: StrictFloat
    sensor_10: StrictFloat
    sensor_11: StrictFloat
    sensor_12: StrictFloat
    sensor_13: StrictFloat
    sensor_14: StrictFloat
    sensor_15: StrictFloat
    sensor_16: StrictFloat
    sensor_17: StrictFloat
    sensor_18: StrictFloat
    sensor_19: StrictFloat
    sensor_20: StrictFloat
    sensor_21: StrictFloat

    @field_validator(
        "op_setting_1",
        "op_setting_2",
        "op_setting_3",
        *[f"sensor_{index}" for index in range(1, 22)],
    )
    @classmethod
    def validate_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("observation values must be finite")
        return value


class EngineHistoryRequest(BaseModel):
    """Prediction request for one engine history."""

    model_config = ConfigDict(extra="forbid")

    unit_id: str = Field(min_length=1, max_length=100)
    observations: list[Observation] = Field(min_length=20, max_length=2000)

    @model_validator(mode="after")
    def validate_ordered_cycles(self) -> EngineHistoryRequest:
        cycles = [observation.cycle for observation in self.observations]
        if cycles != sorted(cycles) or len(set(cycles)) != len(cycles):
            raise ValueError("observations must have strictly increasing cycles")
        return self


class BatchPredictionRequest(BaseModel):
    """Batch prediction request limited to 100 engines."""

    model_config = ConfigDict(extra="forbid")

    histories: list[EngineHistoryRequest] = Field(min_length=1, max_length=100)


class PredictionResponse(BaseModel):
    """Fixed single-prediction response contract."""

    unit_id: str
    predicted_rul: float
    risk_level: Literal["critical", "warning", "healthy"]
    model_version: str
    feature_schema_version: Literal["v1"]
    request_id: str
    predicted_at: datetime


class BatchPredictionResponse(BaseModel):
    """Batch response with one fixed prediction per requested engine."""

    predictions: list[PredictionResponse]
    request_id: str


class HealthResponse(BaseModel):
    status: Literal["healthy"]


class ReadyResponse(BaseModel):
    status: Literal["ready"]
    model_version: str
    feature_schema_version: Literal["v1"]


class ModelInfoResponse(BaseModel):
    model_version: str
    feature_schema_version: Literal["v1"]
    model_type: str
    target: str
