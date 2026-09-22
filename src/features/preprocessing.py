"""Development-only feature filtering, scaling, and serialization."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

from src.features.build_features import FEATURE_COLUMNS


class FeaturePreprocessor:
    """Fit feature transformations on development data and reuse them safely."""

    def __init__(self, variance_threshold: float = 1e-6, scale: bool = False) -> None:
        if variance_threshold < 0:
            raise ValueError("variance_threshold must be non-negative")
        self.variance_threshold = variance_threshold
        self.scale = scale
        self._variance_filter = VarianceThreshold(threshold=variance_threshold)
        self._scaler = StandardScaler() if scale else None
        self.feature_names_: tuple[str, ...] | None = None
        self.selected_feature_names_: tuple[str, ...] | None = None

    def fit(self, features: pd.DataFrame) -> "FeaturePreprocessor":
        """Fit filtering and optional scaling using development features only."""
        self._validate_input_columns(features)
        self.feature_names_ = tuple(features.columns)
        self._variance_filter.fit(features)
        selected = self._variance_filter.get_support()
        self.selected_feature_names_ = tuple(
            name for name, keep in zip(self.feature_names_, selected) if keep
        )
        filtered = self._variance_filter.transform(features)
        if self._scaler is not None:
            self._scaler.fit(filtered)
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Transform features using the already-fitted development contract."""
        if self.feature_names_ is None or self.selected_feature_names_ is None:
            raise RuntimeError("fit must be called before transform")
        self._validate_input_columns(features)
        transformed = self._variance_filter.transform(features)
        if self._scaler is not None:
            transformed = self._scaler.transform(transformed)
        return pd.DataFrame(
            transformed,
            columns=self.selected_feature_names_,
            index=features.index,
        )

    def fit_transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Fit on development features and return their transformed values."""
        return self.fit(features).transform(features)

    def save(self, path: Path) -> None:
        """Serialize the fitted preprocessor."""
        if self.feature_names_ is None:
            raise RuntimeError("fit must be called before save")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "FeaturePreprocessor":
        """Load a serialized fitted preprocessor."""
        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError("Serialized object is not a FeaturePreprocessor")
        return loaded

    @staticmethod
    def _validate_input_columns(features: pd.DataFrame) -> None:
        if tuple(features.columns) != FEATURE_COLUMNS:
            raise ValueError("Feature columns do not match the v1 feature contract.")
