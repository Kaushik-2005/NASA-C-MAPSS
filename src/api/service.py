"""Champion model loading and online/offline feature-parity service."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

import joblib
import pandas as pd

from src.data.schema import EXPECTED_COLUMNS
from src.features.build_features import FEATURE_SCHEMA_VERSION, build_features
from src.features.preprocessing import FeaturePreprocessor


class ModelService:
    """Load the champion model and preprocessor once for the app lifetime."""

    def __init__(
        self,
        model_path: Path | None = None,
        preprocessor_path: Path | None = None,
        model_version: str = "EngineGuardRUL@champion",
    ) -> None:
        project_root = Path(__file__).resolve().parents[2]
        model_path = model_path or project_root / "models/xgboost_candidate_v1.joblib"
        preprocessor_path = (
            preprocessor_path or project_root / "models/feature_preprocessor_v1.joblib"
        )
        self.model = joblib.load(model_path)
        self.preprocessor = FeaturePreprocessor.load(preprocessor_path)
        self.model_version = model_version
        self.feature_schema_version: Literal["v1"] = cast(Literal["v1"], FEATURE_SCHEMA_VERSION)

    def predict(self, unit_id: str, observations: list[dict[str, object]]) -> float:
        history = pd.DataFrame(observations)
        history.insert(0, "unit_id", unit_id)
        history = history.loc[:, list(EXPECTED_COLUMNS)]
        raw_features = build_features(history)
        transformed = self.preprocessor.transform(raw_features)
        raw_prediction = float(self.model.predict(transformed)[0])
        return float(max(0.0, min(125.0, raw_prediction)))


def risk_level(predicted_rul: float) -> Literal["critical", "warning", "healthy"]:
    """Map bounded RUL to deterministic maintenance risk."""
    if predicted_rul <= 30:
        return "critical"
    if predicted_rul <= 60:
        return "warning"
    return "healthy"
