"""Tree regressors and the fixed Module 8 grouped-search contract."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from xgboost import XGBRegressor

RANDOM_SEED = 42
CV_SPLITS = 5
XGBOOST_SEARCH_ITERATIONS = 30

XGBOOST_SEARCH_SPACE: dict[str, list[Any]] = {
    "n_estimators": [100, 200, 400, 600],
    "max_depth": [2, 3, 4, 6],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
    "min_child_weight": [1, 3, 5],
    "reg_alpha": [0.0, 0.01, 0.1],
    "reg_lambda": [1.0, 5.0, 10.0],
}


def build_random_forest() -> RandomForestRegressor:
    """Build the reproducible Random Forest candidate."""
    return RandomForestRegressor(
        n_estimators=400,
        random_state=RANDOM_SEED,
        n_jobs=1,
    )


def build_xgboost() -> XGBRegressor:
    """Build the reproducible XGBoost candidate before hyperparameter tuning."""
    return XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_SEED,
        n_jobs=1,
        tree_method="hist",
        eval_metric="rmse",
    )


def build_grouped_cv() -> GroupKFold:
    """Return the required five-fold engine-grouped cross-validator."""
    return GroupKFold(n_splits=CV_SPLITS)


def build_xgboost_search() -> RandomizedSearchCV:
    """Build the exact 30-configuration XGBoost search."""
    return RandomizedSearchCV(
        estimator=build_xgboost(),
        param_distributions=XGBOOST_SEARCH_SPACE,
        n_iter=XGBOOST_SEARCH_ITERATIONS,
        scoring="neg_root_mean_squared_error",
        cv=build_grouped_cv(),
        random_state=RANDOM_SEED,
        n_jobs=1,
        refit=True,
    )


def validate_group_labels(groups: np.ndarray, expected_samples: int) -> None:
    """Reject missing or malformed engine-group labels before cross-validation."""
    group_values = np.asarray(groups)

    if group_values.ndim != 1:
        raise ValueError("groups must be one-dimensional")

    if group_values.shape[0] != expected_samples:
        raise ValueError("groups must have one value per training sample")

    if not np.isfinite(group_values).all():
        raise ValueError("groups contains non-finite values")

    if np.unique(group_values).size < CV_SPLITS:
        raise ValueError("groups must contain at least five unique engines")
