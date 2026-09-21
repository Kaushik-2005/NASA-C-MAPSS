from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def _validate_inputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    true_values = np.asarray(y_true, dtype=float)
    predicted_values = np.asarray(y_pred, dtype=float)

    if true_values.shape != predicted_values.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    if true_values.size == 0:
        raise ValueError("Metric inputs cannot be empty")

    if not np.isfinite(true_values).all():
        raise ValueError("y_true contains non-finite values")

    if not np.isfinite(predicted_values).all():
        raise ValueError("y_pred contains non-finite values")

    return true_values, predicted_values


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_values, predicted_values = _validate_inputs(y_true, y_pred)
    return float(
        np.sqrt(mean_squared_error(true_values, predicted_values))
    )


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_values, predicted_values = _validate_inputs(y_true, y_pred)
    return float(mean_absolute_error(true_values, predicted_values))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_values, predicted_values = _validate_inputs(y_true, y_pred)
    return float(r2_score(true_values, predicted_values))


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r_squared": r_squared(y_true, y_pred),
    }


def _validate_binary_inputs(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    raw_labels = np.asarray(y_true)
    scores = np.asarray(y_score, dtype=float)

    if raw_labels.ndim != 1 or scores.ndim != 1:
        raise ValueError("Classification inputs must be one-dimensional")

    if raw_labels.shape != scores.shape:
        raise ValueError("y_true and y_score must have the same shape")

    if raw_labels.size == 0:
        raise ValueError("Classification inputs cannot be empty")

    if not np.isfinite(raw_labels).all() or not np.isin(raw_labels, [0, 1]).all():
        raise ValueError("y_true must contain only binary labels 0 and 1")

    labels = raw_labels.astype(int)

    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("y_score must contain finite probabilities between 0 and 1")

    if np.unique(labels).size < 2:
        raise ValueError("y_true must contain both binary classes")

    return labels, scores


def classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | list[list[int]]]:
    """Calculate thresholded and ranking metrics for binary classification."""
    if not 0 < threshold < 1:
        raise ValueError("threshold must be between 0 and 1")

    labels, scores = _validate_binary_inputs(y_true, y_score)
    predictions = (scores >= threshold).astype(int)

    matrix = confusion_matrix(labels, predictions, labels=[0, 1])

    return {
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "threshold": float(threshold),
        "confusion_matrix": matrix.astype(int).tolist(),
    }
