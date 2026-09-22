"""Reproducible development training and MLflow registration entry point."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.data.ingest import load_fd001_file
from src.data.labels import add_rul_labels
from src.data.samples import generate_training_samples, generate_validation_samples
from src.data.splits import build_engine_manifests
from src.evaluation.metrics import regression_metrics
from src.features.build_features import build_feature_matrix
from src.features.preprocessing import FeaturePreprocessor
from src.training.trees import build_xgboost

REGISTRY_NAME = "EngineGuardRUL"
EXPERIMENT_NAME = "engineguard-reproducible-training"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return result.stdout.strip()


def _prepare_training_data(
    train_path: Path,
    preprocessor_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, FeaturePreprocessor]:
    labeled_train = add_rul_labels(load_fd001_file(train_path))
    manifests = build_engine_manifests(labeled_train["unit_id"].unique())
    development_samples = generate_training_samples(
        labeled_train,
        manifests["development"],
    )
    validation_samples = generate_validation_samples(
        labeled_train,
        manifests["validation"],
    ).samples

    development_raw = build_feature_matrix(labeled_train, development_samples)
    validation_raw = build_feature_matrix(labeled_train, validation_samples)
    preprocessor = FeaturePreprocessor.load(preprocessor_path)

    return (
        preprocessor.transform(development_raw).to_numpy(dtype=float),
        preprocessor.transform(validation_raw).to_numpy(dtype=float),
        development_samples["target_rul"].to_numpy(dtype=float),
        validation_samples["target_rul"].to_numpy(dtype=float),
        preprocessor,
    )


def _register_model(run_id: str, model_uri: str) -> dict[str, str]:
    client = MlflowClient()
    model_version = mlflow.register_model(model_uri, REGISTRY_NAME)
    version = str(model_version.version)
    client.set_registered_model_alias(REGISTRY_NAME, "candidate", version)

    try:
        client.get_model_version_by_alias(REGISTRY_NAME, "champion")
        champion_status = "preserved"
    except MlflowException:
        client.set_registered_model_alias(REGISTRY_NAME, "champion", version)
        champion_status = "initialized"

    return {
        "run_id": run_id,
        "registered_model": REGISTRY_NAME,
        "model_version": version,
        "candidate_alias": f"{REGISTRY_NAME}@candidate",
        "champion_alias": f"{REGISTRY_NAME}@champion",
        "champion_status": champion_status,
    }


def run_reproducible_training(
    train_path: Path = Path("data/raw/train_FD001.txt"),
    candidate_path: Path = Path("configs/xgboost-candidate-v1.json"),
    preprocessor_path: Path = Path("models/feature_preprocessor_v1.joblib"),
    model_path: Path = Path("models/xgboost_candidate_v1.joblib"),
    lineage_path: Path = Path("reports/training-lineage.json"),
) -> dict[str, Any]:
    """Train and register the frozen candidate without using notebooks or test labels."""
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    (
        X_train,
        X_validation,
        y_train,
        y_validation,
        _preprocessor,
    ) = _prepare_training_data(train_path, preprocessor_path)

    model = build_xgboost().set_params(**candidate["best_params"])
    model.fit(X_train, y_train)
    validation_predictions = np.clip(model.predict(X_validation), 0, 125)
    metrics = regression_metrics(y_validation, validation_predictions)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    config_hash = _sha256(candidate_path)
    raw_hash = _sha256(train_path)
    dvc_lock_path = Path("dvc.lock")
    lineage: dict[str, Any] = {
        "dataset": "NASA C-MAPSS FD001",
        "data_partition": "development training and fixed validation samples",
        "raw_train_sha256": raw_hash,
        "dvc_lock_sha256": _sha256(dvc_lock_path) if dvc_lock_path.exists() else "unavailable",
        "git_revision": _git_revision(),
        "feature_schema_version": candidate["feature_schema_version"],
        "candidate_config_sha256": config_hash,
        "candidate_config": candidate,
        "training_samples": len(X_train),
        "validation_samples": len(X_validation),
        "validation_metrics": metrics,
        "model_artifact": str(model_path),
        "preprocessor_artifact": str(preprocessor_path),
    }

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="fd001-reproducible-candidate") as run:
        mlflow.log_params(
            {
                "dataset": lineage["dataset"],
                "feature_schema_version": lineage["feature_schema_version"],
                "random_seed": candidate["random_seed"],
                "n_estimators": candidate["best_params"]["n_estimators"],
                "max_depth": candidate["best_params"]["max_depth"],
                "learning_rate": candidate["best_params"]["learning_rate"],
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(candidate_path))
        mlflow.log_artifact(str(preprocessor_path))
        mlflow.log_artifact(str(model_path))
        model_info = mlflow.sklearn.log_model(model, "model")
        registration = _register_model(run.info.run_id, model_info.model_uri)

    lineage["mlflow"] = registration
    lineage_path.parent.mkdir(parents=True, exist_ok=True)
    lineage_path.write_text(json.dumps(lineage, indent=2) + "\n", encoding="utf-8")
    return lineage


if __name__ == "__main__":
    result = run_reproducible_training()
    print(f"Registered {result['mlflow']['registered_model']} version {result['mlflow']['model_version']}")
