"""Verify the NumPy foundations against scikit-learn on FD001 development data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error

from src.data.ingest import load_fd001_file
from src.data.labels import add_rul_labels
from src.data.splits import build_engine_manifests
from src.foundations.linear_regression import LinearRegressionGD
from src.foundations.logistic_regression import LogisticRegressionGD

SEED = 42
FEATURE_COLUMNS = ["cycle", "sensor_2"]
REPORT_PATH = Path("reports/foundations-fd001.json")


def _standardize(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=0)
    standard_deviation = values.std(axis=0)
    return (values - mean) / np.where(standard_deviation == 0, 1.0, standard_deviation)


def run_fd001_verification(
    train_path: Path = Path("data/raw/train_FD001.txt"),
) -> dict[str, Any]:
    """Run a development-only comparison and return measured evidence."""
    np.random.seed(SEED)
    labeled = add_rul_labels(load_fd001_file(train_path))
    manifests = build_engine_manifests(labeled["unit_id"].unique())
    development = labeled[labeled["unit_id"].isin(manifests["development"])].copy()
    X = _standardize(development[FEATURE_COLUMNS].to_numpy(dtype=float))
    y_rul = development["target_rul"].to_numpy(dtype=float)
    y_failure = development["failure_within_30"].to_numpy(dtype=int)

    numpy_linear = LinearRegressionGD(learning_rate=0.05, epochs=1_000).fit(X, y_rul)
    sklearn_linear = LinearRegression().fit(X, y_rul)
    numpy_logistic = LogisticRegressionGD(learning_rate=0.1, epochs=1_000).fit(X, y_failure)
    sklearn_logistic = LogisticRegression(max_iter=2_000, random_state=SEED).fit(X, y_failure)

    learning_rates: dict[str, float] = {}
    for rate in (0.01, 0.05, 0.2):
        model = LinearRegressionGD(learning_rate=rate, epochs=300).fit(X, y_rul)
        learning_rates[str(rate)] = float(model.loss_history_[-1])

    regularization: dict[str, float] = {}
    for penalty in (0.0, 0.1, 1.0):
        model = LinearRegressionGD(learning_rate=0.05, epochs=1_000, l2=penalty).fit(X, y_rul)
        regularization[str(penalty)] = float(mean_squared_error(y_rul, model.predict(X)))

    report: dict[str, Any] = {
        "dataset": "NASA C-MAPSS FD001",
        "partition": "development engines only",
        "engine_count": len(manifests["development"]),
        "row_count": len(development),
        "features": FEATURE_COLUMNS,
        "seed": SEED,
        "regression": {
            "numpy_rmse": float(mean_squared_error(y_rul, numpy_linear.predict(X)) ** 0.5),
            "sklearn_rmse": float(mean_squared_error(y_rul, sklearn_linear.predict(X)) ** 0.5),
            "numpy_final_mse": float(numpy_linear.loss_history_[-1]),
            "numpy_loss_decreased": bool(numpy_linear.loss_history_[-1] < numpy_linear.loss_history_[0]),
        },
        "classification": {
            "numpy_accuracy": float(accuracy_score(y_failure, numpy_logistic.predict(X))),
            "sklearn_accuracy": float(accuracy_score(y_failure, sklearn_logistic.predict(X))),
            "numpy_final_log_loss": float(numpy_logistic.loss_history_[-1]),
            "numpy_loss_decreased": bool(numpy_logistic.loss_history_[-1] < numpy_logistic.loss_history_[0]),
        },
        "learning_rate_final_mse": learning_rates,
        "l2_training_mse": regularization,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run_fd001_verification(), indent=2))
