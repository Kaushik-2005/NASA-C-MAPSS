"""Train and benchmark Module 7 baseline models on FD001 development data."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.data.ingest import load_fd001_file
from src.data.labels import add_rul_labels
from src.data.samples import generate_training_samples, generate_validation_samples
from src.data.splits import build_engine_manifests
from src.evaluation.metrics import classification_metrics, regression_metrics
from src.features.build_features import build_feature_matrix
from src.features.preprocessing import FeaturePreprocessor
from src.training.baselines import (
    LogisticFailureClassifier,
    MedianRULRegressor,
    RidgeRULRegressor,
)


def _timed_fit(model: Any, X: np.ndarray, y: np.ndarray) -> float:
    start = time.perf_counter()
    model.fit(X, y)
    return (time.perf_counter() - start) * 1000


def _timed_predict(model: Any, X: np.ndarray) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    predictions = model.predict(X)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return np.asarray(predictions), elapsed_ms


def _serialize_model(model: Any, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path.stat().st_size


def _regression_result(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    artifact_path: Path,
) -> dict[str, Any]:
    training_time_ms = _timed_fit(model, X_train, y_train)
    predictions, inference_time_ms = _timed_predict(model, X_validation)
    metrics = regression_metrics(y_validation, predictions)
    model_size_bytes = _serialize_model(model, artifact_path)
    return {
        "metrics": metrics,
        "training_time_ms": training_time_ms,
        "inference_time_ms": inference_time_ms,
        "model_size_bytes": model_size_bytes,
        "artifact": str(artifact_path),
    }


def _classification_result(
    model: LogisticFailureClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    artifact_path: Path,
) -> dict[str, Any]:
    training_time_ms = _timed_fit(model, X_train, y_train)
    start = time.perf_counter()
    scores = model.predict_proba(X_validation)[:, 1]
    inference_time_ms = (time.perf_counter() - start) * 1000
    metrics_default = classification_metrics(y_validation, scores, threshold=0.5)
    metrics_recall = classification_metrics(y_validation, scores, threshold=0.3)
    model_size_bytes = _serialize_model(model, artifact_path)
    return {
        "default_threshold": metrics_default,
        "recall_oriented_threshold": metrics_recall,
        "training_time_ms": training_time_ms,
        "inference_time_ms": inference_time_ms,
        "model_size_bytes": model_size_bytes,
        "artifact": str(artifact_path),
    }


def _log_mlflow_results(results: dict[str, Any]) -> dict[str, str]:
    """Log measured baseline results when the declared MLflow dependency exists."""
    try:
        import mlflow
    except ImportError:
        return {
            "status": "not configured",
            "reason": "mlflow is not installed in the project environment",
        }

    mlflow.set_experiment("engineguard-baselines")
    with mlflow.start_run(run_name="fd001-module-7-baselines"):
        mlflow.log_params(
            {
                "dataset": results["dataset"],
                "feature_schema_version": results["feature_schema_version"],
                "train_partition": results["train_partition"],
                "validation_partition": results["validation_partition"],
                "standardized_feature_count": results["standardized_feature_count"],
            }
        )
        for name, result in results["regression"].items():
            mlflow.log_metrics(
                {
                    f"{name}_rmse": result["metrics"]["rmse"],
                    f"{name}_mae": result["metrics"]["mae"],
                    f"{name}_r_squared": result["metrics"]["r_squared"],
                    f"{name}_training_time_ms": result["training_time_ms"],
                    f"{name}_inference_time_ms": result["inference_time_ms"],
                    f"{name}_model_size_bytes": result["model_size_bytes"],
                }
            )
            mlflow.log_artifact(result["artifact"])

        logistic = results["classification"]["logistic"]
        mlflow.log_metrics(
            {
                "logistic_precision_default": logistic["default_threshold"]["precision"],
                "logistic_recall_default": logistic["default_threshold"]["recall"],
                "logistic_f1_default": logistic["default_threshold"]["f1"],
                "logistic_roc_auc": logistic["default_threshold"]["roc_auc"],
                "logistic_pr_auc": logistic["default_threshold"]["pr_auc"],
                "logistic_training_time_ms": logistic["training_time_ms"],
                "logistic_inference_time_ms": logistic["inference_time_ms"],
                "logistic_model_size_bytes": logistic["model_size_bytes"],
            }
        )
        mlflow.log_artifact(logistic["artifact"])

    return {
        "status": "logged",
        "experiment": "engineguard-baselines",
    }


def run_baseline_benchmark(
    train_path: Path = Path("data/raw/train_FD001.txt"),
    model_dir: Path = Path("models"),
    report_path: Path = Path("reports/baseline-results.md"),
    json_path: Path = Path("reports/baseline-results.json"),
) -> dict[str, Any]:
    """Train baselines using development data and evaluate fixed validation samples."""
    labeled_train = add_rul_labels(load_fd001_file(train_path))
    manifests = build_engine_manifests(labeled_train["unit_id"].unique())
    development_samples = generate_training_samples(
        labeled_train,
        manifests["development"],
    )
    validation_result = generate_validation_samples(
        labeled_train,
        manifests["validation"],
    )
    validation_samples = validation_result.samples

    development_raw = build_feature_matrix(labeled_train, development_samples)
    validation_raw = build_feature_matrix(labeled_train, validation_samples)

    preprocessor = FeaturePreprocessor(scale=True)
    development_features = preprocessor.fit_transform(development_raw)
    validation_features = preprocessor.transform(validation_raw)

    X_train = development_features.to_numpy(dtype=float)
    X_validation = validation_features.to_numpy(dtype=float)
    y_train_rul = development_samples["target_rul"].to_numpy(dtype=float)
    y_validation_rul = validation_samples["target_rul"].to_numpy(dtype=float)
    y_train_failure = development_samples["failure_within_30"].to_numpy(dtype=int)
    y_validation_failure = validation_samples["failure_within_30"].to_numpy(dtype=int)

    results: dict[str, Any] = {
        "dataset": "NASA C-MAPSS FD001",
        "feature_schema_version": "v1",
        "train_partition": "development (80 engines)",
        "validation_partition": "fixed validation samples (20 engines, 4 cutoffs)",
        "development_samples": len(development_samples),
        "validation_samples": len(validation_samples),
        "validation_cutoffs_requested": validation_result.requested_cutoffs,
        "validation_cutoffs_deduplicated": validation_result.deduplicated_cutoffs,
        "standardized_feature_count": int(X_train.shape[1]),
        "regression": {},
        "classification": {},
        "mlflow": {
            "status": "not configured",
            "reason": "mlflow is not installed in the project environment",
        },
    }

    results["regression"]["median"] = _regression_result(
        MedianRULRegressor(),
        X_train,
        y_train_rul,
        X_validation,
        y_validation_rul,
        model_dir / "median_rul_v1.joblib",
    )
    results["regression"]["ridge"] = _regression_result(
        RidgeRULRegressor(),
        X_train,
        y_train_rul,
        X_validation,
        y_validation_rul,
        model_dir / "ridge_rul_v1.joblib",
    )
    results["classification"]["logistic"] = _classification_result(
        LogisticFailureClassifier(),
        X_train,
        y_train_failure,
        X_validation,
        y_validation_failure,
        model_dir / "logistic_failure_v1.joblib",
    )
    results["mlflow"] = _log_mlflow_results(results)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    _write_markdown_report(results, report_path)
    return results


def _write_markdown_report(results: dict[str, Any], report_path: Path) -> None:
    regression = results["regression"]
    classification = results["classification"]["logistic"]
    lines = [
        "# Module 7 Baseline Results",
        "",
        "These results use only the FD001 development engines for training and the fixed 20-engine validation partition for evaluation. Official test labels were not used.",
        "",
        f"- Development samples: {results['development_samples']}",
        f"- Validation samples: {results['validation_samples']}",
        f"- Standardized selected features: {results['standardized_feature_count']}",
        "",
        "## Regression",
        "",
        "| Model | RMSE | MAE | R-squared | Train ms | Inference ms | Size bytes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, result in regression.items():
        metrics = result["metrics"]
        lines.append(
            f"| {name} | {metrics['rmse']:.4f} | {metrics['mae']:.4f} | "
            f"{metrics['r_squared']:.4f} | {result['training_time_ms']:.3f} | "
            f"{result['inference_time_ms']:.3f} | {result['model_size_bytes']} |"
        )

    default = classification["default_threshold"]
    recall = classification["recall_oriented_threshold"]
    lines.extend(
        [
            "",
            "## Logistic classification",
            "",
            "The classifier predicts `failure_within_30`. PR-AUC is reported as average precision.",
            "",
            "| Threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC | Confusion matrix |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            f"| {default['threshold']:.2f} | {default['precision']:.4f} | {default['recall']:.4f} | {default['f1']:.4f} | {default['roc_auc']:.4f} | {default['pr_auc']:.4f} | `{default['confusion_matrix']}` |",
            f"| {recall['threshold']:.2f} | {recall['precision']:.4f} | {recall['recall']:.4f} | {recall['f1']:.4f} | {recall['roc_auc']:.4f} | {recall['pr_auc']:.4f} | `{recall['confusion_matrix']}` |",
            "",
            f"- Logistic training time: {classification['training_time_ms']:.3f} ms",
            f"- Logistic inference time: {classification['inference_time_ms']:.3f} ms",
            f"- Logistic serialized size: {classification['model_size_bytes']} bytes",
            "",
            "## Limitations",
            "",
            "- These are fixed validation results, not official FD001 test results.",
            f"- MLflow status: {results['mlflow']['status']}.",
            "- Timing is a local single-run measurement and is not a production latency claim.",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run_baseline_benchmark()
    print("Baseline benchmark written")
