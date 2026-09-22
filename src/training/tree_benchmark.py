"""Train and compare Random Forest and tuned XGBoost on FD001 validation data."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np

from src.data.ingest import load_fd001_file
from src.data.labels import add_rul_labels
from src.data.samples import generate_training_samples, generate_validation_samples
from src.data.splits import build_engine_manifests
from src.evaluation.metrics import regression_metrics
from src.features.build_features import build_feature_matrix
from src.features.preprocessing import FeaturePreprocessor
from src.training.trees import build_random_forest, build_xgboost_search


def _fit_and_measure(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    artifact_path: Path,
) -> dict[str, Any]:
    start = time.perf_counter()
    model.fit(X_train, y_train)
    training_time_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    predictions = np.asarray(model.predict(X_validation), dtype=float)
    inference_time_ms = (time.perf_counter() - start) * 1000

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact_path)

    return {
        "metrics": regression_metrics(y_validation, predictions),
        "training_time_ms": training_time_ms,
        "inference_time_ms": inference_time_ms,
        "model_size_bytes": artifact_path.stat().st_size,
        "artifact": str(artifact_path),
    }


def _prepare_data(
    train_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
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

    development_raw = build_feature_matrix(labeled_train, development_samples)
    validation_raw = build_feature_matrix(labeled_train, validation_result.samples)
    preprocessor = FeaturePreprocessor(scale=False)
    development_features = preprocessor.fit_transform(development_raw)
    validation_features = preprocessor.transform(validation_raw)

    return (
        development_features.to_numpy(dtype=float),
        validation_features.to_numpy(dtype=float),
        development_samples["target_rul"].to_numpy(dtype=float),
        validation_result.samples["target_rul"].to_numpy(dtype=float),
        development_samples["unit_id"].to_numpy(dtype=int),
        validation_result.deduplicated_cutoffs,
    )


def _log_mlflow(
    results: dict[str, Any],
    search: Any,
    candidate_path: Path,
) -> str:
    mlflow.set_experiment("engineguard-tree-models")
    with mlflow.start_run(run_name="fd001-random-forest-xgboost"):
        mlflow.log_params(
            {
                "dataset": "NASA C-MAPSS FD001",
                "feature_schema_version": "v1",
                "cv": "GroupKFold(n_splits=5)",
                "search_iterations": search.n_iter,
                "search_scoring": search.scoring,
                "random_seed": 42,
            }
        )
        for name, result in results.items():
            metrics = result["metrics"]
            mlflow.log_metrics(
                {
                    f"{name}_rmse": metrics["rmse"],
                    f"{name}_mae": metrics["mae"],
                    f"{name}_r_squared": metrics["r_squared"],
                    f"{name}_training_time_ms": result["training_time_ms"],
                    f"{name}_inference_time_ms": result["inference_time_ms"],
                    f"{name}_model_size_bytes": result["model_size_bytes"],
                }
            )
            mlflow.log_artifact(result["artifact"])
        mlflow.log_metric("xgboost_best_cv_rmse", -float(search.best_score_))
        mlflow.log_artifact(str(candidate_path))
    return "logged"


def _write_report(
    results: dict[str, Any],
    search: Any,
    metadata: dict[str, Any],
    report_path: Path,
) -> None:
    lines = [
        "# Module 8 Tree Model Comparison",
        "",
        "Training used only the 80 FD001 development engines. Hyperparameter selection used five-fold `GroupKFold` by engine. The fixed 20-engine validation samples were used once for the comparison below. Official test labels were not used.",
        "",
        f"- Development samples: {metadata['development_samples']}",
        f"- Validation samples: {metadata['validation_samples']}",
        f"- Selected features: {metadata['selected_feature_count']}",
        f"- XGBoost configurations evaluated: {search.n_iter}",
        f"- Best grouped-CV RMSE: {-float(search.best_score_):.4f}",
        "",
        "## Validation comparison",
        "",
        "| Model | RMSE | MAE | R-squared | Train ms | Inference ms | Size bytes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, result in results.items():
        metrics = result["metrics"]
        lines.append(
            f"| {name} | {metrics['rmse']:.4f} | {metrics['mae']:.4f} | "
            f"{metrics['r_squared']:.4f} | {result['training_time_ms']:.3f} | "
            f"{result['inference_time_ms']:.3f} | {result['model_size_bytes']} |"
        )
    lines.extend(
        [
            "",
            "## Selected XGBoost configuration",
            "",
            "```json",
            json.dumps(search.best_params_, indent=2),
            "```",
            "",
            "## Interpretation",
            "",
            "The grouped-CV score selects the XGBoost configuration without using the fixed validation labels. Validation metrics are comparison evidence, not official final-test evidence.",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_tree_benchmark(
    train_path: Path = Path("data/raw/train_FD001.txt"),
    model_dir: Path = Path("models"),
    report_path: Path = Path("reports/tree-model-comparison.md"),
    json_path: Path = Path("reports/tree-model-comparison.json"),
    candidate_path: Path = Path("configs/xgboost-candidate-v1.json"),
) -> dict[str, Any]:
    """Run the fixed Random Forest and 30-trial XGBoost benchmark."""
    (
        X_train,
        X_validation,
        y_train,
        y_validation,
        groups,
        deduplicated_cutoffs,
    ) = _prepare_data(train_path)

    metadata = {
        "dataset": "NASA C-MAPSS FD001",
        "feature_schema_version": "v1",
        "development_samples": int(X_train.shape[0]),
        "validation_samples": int(X_validation.shape[0]),
        "selected_feature_count": int(X_train.shape[1]),
        "validation_cutoffs_deduplicated": deduplicated_cutoffs,
    }

    results = {
        "random_forest": _fit_and_measure(
            build_random_forest(),
            X_train,
            y_train,
            X_validation,
            y_validation,
            model_dir / "random_forest_rul_v1.joblib",
        ),
    }

    search = build_xgboost_search()
    start = time.perf_counter()
    search.fit(X_train, y_train, groups=groups)
    search_time_ms = (time.perf_counter() - start) * 1000
    xgb_result = _fit_and_measure(
        search.best_estimator_,
        X_train,
        y_train,
        X_validation,
        y_validation,
        model_dir / "xgboost_rul_v1.joblib",
    )
    xgb_result["search_time_ms"] = search_time_ms
    xgb_result["best_cv_rmse"] = -float(search.best_score_)
    results["xgboost"] = xgb_result

    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.write_text(
        json.dumps(
            {
                "feature_schema_version": "v1",
                "random_seed": 42,
                "search_iterations": search.n_iter,
                "cv": "GroupKFold(n_splits=5)",
                "scoring": search.scoring,
                "best_cv_rmse": -float(search.best_score_),
                "best_params": search.best_params_,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    metadata["results"] = results
    metadata["best_params"] = search.best_params_
    metadata["mlflow"] = _log_mlflow(results, search, candidate_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    _write_report(results, search, metadata, report_path)
    return metadata


if __name__ == "__main__":
    run_tree_benchmark()
    print("Tree model comparison written")
