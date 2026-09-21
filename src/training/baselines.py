from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.linear_model import LogisticRegression, Ridge


def _validate_features(X: np.ndarray) -> np.ndarray:
    features = np.asarray(X, dtype=float)

    if features.ndim != 2:
        raise ValueError("X must be a two-dimensional feature matrix")

    if features.shape[0] == 0:
        raise ValueError("X cannot be empty")

    if not np.isfinite(features).all():
        raise ValueError("X contains non-finite values")

    return features


def _validate_target(y: np.ndarray, sample_count: int) -> np.ndarray:
    target = np.asarray(y, dtype=float)

    if target.ndim != 1:
        raise ValueError("y must be one-dimensional")

    if target.shape[0] != sample_count:
        raise ValueError("X and y must contain the same number of samples")

    if not np.isfinite(target).all():
        raise ValueError("y contains non-finite values")

    return target


def _validate_binary_target(y: np.ndarray, sample_count: int) -> np.ndarray:
    raw_target = np.asarray(y)

    if raw_target.ndim != 1:
        raise ValueError("y must be one-dimensional")

    if raw_target.shape[0] != sample_count:
        raise ValueError("X and y must contain the same number of samples")

    if not np.isfinite(raw_target).all() or not np.isin(raw_target, [0, 1]).all():
        raise ValueError("y must contain only binary labels 0 and 1")

    target = raw_target.astype(int)

    if np.unique(target).size < 2:
        raise ValueError("y must contain both binary classes")

    return target


class MedianRULRegressor(
    BaseEstimator,
    RegressorMixin,
):
    """Predict the median target RUL for every sample."""

    def __init__(self) -> None:
        self.median_rul_: float | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> MedianRULRegressor:
        del X

        target = np.asarray(y, dtype=float)

        if target.ndim != 1:
            raise ValueError("y must be one-dimensional")

        if target.size == 0:
            raise ValueError("y cannot be empty")

        if not np.isfinite(target).all():
            raise ValueError("y contains non-finite values")

        self.median_rul_ = float(np.median(target))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.median_rul_ is None:
            raise RuntimeError("The model must be fitted before prediction")

        sample_count = np.asarray(X).shape[0]

        return np.full(
            shape=sample_count,
            fill_value=self.median_rul_,
            dtype=float,
        )


class RidgeRULRegressor(
    BaseEstimator,
    RegressorMixin,
):
    """Ridge RUL regressor for standardized EngineGuard features."""

    def __init__(self, alpha: float = 1.0) -> None:
        if alpha <= 0:
            raise ValueError("alpha must be greater than zero")

        self.alpha = alpha
        self._model = Ridge(alpha=alpha)
        self.is_fitted_ = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> RidgeRULRegressor:
        features = _validate_features(X)
        target = _validate_target(y, features.shape[0])

        self._model.fit(features, target)
        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("The model must be fitted before prediction")

        features = _validate_features(X)
        return np.asarray(self._model.predict(features), dtype=float)


class LogisticFailureClassifier(
    BaseEstimator,
    ClassifierMixin,
):
    """Logistic Regression learning artifact for failure within 30 cycles."""

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42,
    ) -> None:
        if C <= 0:
            raise ValueError("C must be greater than zero")
        if max_iter <= 0:
            raise ValueError("max_iter must be greater than zero")

        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self._model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
        )
        self.is_fitted_ = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> LogisticFailureClassifier:
        features = _validate_features(X)
        target = _validate_binary_target(y, features.shape[0])

        self._model.fit(features, target)
        self.is_fitted_ = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("The model must be fitted before prediction")

        features = _validate_features(X)
        return np.asarray(self._model.predict_proba(features), dtype=float)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("The model must be fitted before prediction")

        features = _validate_features(X)
        return np.asarray(self._model.predict(features), dtype=int)

    def predict_with_threshold(
        self,
        X: np.ndarray,
        threshold: float = 0.5,
    ) -> np.ndarray:
        if not 0 < threshold < 1:
            raise ValueError("threshold must be between 0 and 1")

        probabilities = self.predict_proba(X)[:, 1]
        return (probabilities >= threshold).astype(int)
