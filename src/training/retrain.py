"""Champion-challenger retraining triggers and promotion policy."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
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

RUL_MIN = 0.0
RUL_MAX = 125.0
MIN_RMSE_IMPROVEMENT = 0.03
MAX_VALIDATION_LATENCY_MS = 100.0
PSI_TRIGGER = 0.20
RMSE_TRIGGER_MULTIPLIER = 1.20
REGISTRY_NAME = "EngineGuardRUL"
EXPERIMENT_NAME = "engineguard-retraining"


@dataclass(frozen=True)
class PromotionDecision:
    """Auditable result of evaluating a challenger against the champion."""

    promote: bool
    reasons: tuple[str, ...]
    passed_gates: tuple[str, ...]


def _train_partition(
    labeled: object,
    train_samples: object,
    validation_samples: object,
    *,
    model_path: Path,
    preprocessor_path: Path,
    candidate_params: dict[str, object],
) -> tuple[dict[str, float], np.ndarray, float]:
    """Fit one partition-specific model and score it on fixed validation samples."""
    import pandas as pd

    if not isinstance(labeled, pd.DataFrame) or not isinstance(train_samples, pd.DataFrame):
        raise TypeError("labeled and train_samples must be pandas DataFrames")
    if not isinstance(validation_samples, pd.DataFrame):
        raise TypeError("validation_samples must be a pandas DataFrame")

    train_raw = build_feature_matrix(labeled, train_samples)
    validation_raw = build_feature_matrix(labeled, validation_samples)
    preprocessor = FeaturePreprocessor(scale=False).fit(train_raw)
    X_train = preprocessor.transform(train_raw)
    X_validation = preprocessor.transform(validation_raw)
    model = build_xgboost().set_params(**candidate_params)
    model.fit(X_train, train_samples["target_rul"].to_numpy(dtype=float))

    started = time.perf_counter()
    raw_predictions = np.asarray(model.predict(X_validation), dtype=float)
    prediction_ms = (time.perf_counter() - started) * 1000
    predictions = np.clip(raw_predictions, RUL_MIN, RUL_MAX)
    metrics = regression_metrics(
        validation_samples["target_rul"].to_numpy(dtype=float), predictions
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    preprocessor.save(preprocessor_path)
    return metrics, predictions, prediction_ms


def _p95_single_row_latency(model: object, features: object) -> float:
    """Measure local single-row inference p95 on the fixed validation matrix."""
    import pandas as pd

    if not hasattr(model, "predict") or not isinstance(features, pd.DataFrame):
        raise TypeError("model and features are invalid")
    timings: list[float] = []
    for index in range(len(features)):
        started = time.perf_counter()
        model.predict(features.iloc[[index]])
        timings.append((time.perf_counter() - started) * 1000)
    return float(np.percentile(timings, 95))


def rollback_model(active_path: Path, backup_path: Path) -> None:
    """Restore a champion artifact from an explicitly supplied backup."""
    if not backup_path.exists():
        raise FileNotFoundError(f"Rollback backup not found: {backup_path}")
    active_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(backup_path, active_path)


def _run_promotion_tests() -> tuple[bool, str]:
    """Run the unit suite used as evidence for the promotion gate."""
    command = [sys.executable, "-m", "pytest", "tests/unit", "-q", "--tb=short"]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    output = (completed.stdout + completed.stderr).strip()
    return completed.returncode == 0, output[-4000:]


def _register_retraining_models(
    *,
    candidate: dict[str, Any],
    train_path: Path,
    initial_model: object,
    challenger_model: object,
    initial_preprocessor_path: Path,
    challenger_preprocessor_path: Path,
    initial_metrics: dict[str, float],
    challenger_metrics: dict[str, float],
    initial_model_path: Path,
    challenger_model_path: Path,
) -> dict[str, object]:
    """Log both retraining runs and register the challenger as a candidate."""
    client = MlflowClient()
    mlflow.set_experiment(EXPERIMENT_NAME)
    common_params = {
        "dataset": "NASA C-MAPSS FD001",
        "feature_schema_version": str(candidate["feature_schema_version"]),
        "random_seed": int(candidate["random_seed"]),
        "raw_train_sha256": _sha256(train_path),
        "git_revision": _git_revision(),
    }

    run_records: dict[str, dict[str, str]] = {}
    for label, model, preprocessor_path, model_path, metrics in (
        (
            "initial-champion",
            initial_model,
            initial_preprocessor_path,
            initial_model_path,
            initial_metrics,
        ),
        (
            "incremental-challenger",
            challenger_model,
            challenger_preprocessor_path,
            challenger_model_path,
            challenger_metrics,
        ),
    ):
        with mlflow.start_run(run_name=f"fd001-{label}") as run:
            mlflow.log_params(common_params)
            mlflow.log_params(
                {
                    "training_role": label,
                    "n_estimators": candidate["best_params"]["n_estimators"],
                    "max_depth": candidate["best_params"]["max_depth"],
                    "learning_rate": candidate["best_params"]["learning_rate"],
                }
            )
            mlflow.log_metrics(metrics)
            mlflow.log_dict(candidate, "candidate-config.json")
            dvc_lock_path = Path("dvc.lock")
            if dvc_lock_path.exists():
                mlflow.log_artifact(str(dvc_lock_path))
            mlflow.log_artifact(str(preprocessor_path))
            mlflow.log_artifact(str(model_path))
            model_info = mlflow.sklearn.log_model(model, "model")
            run_records[label] = {
                "run_id": run.info.run_id,
                "model_uri": model_info.model_uri,
            }

    initial_version = str(
        mlflow.register_model(
            run_records["initial-champion"]["model_uri"],
            REGISTRY_NAME,
        ).version
    )
    challenger_version = str(
        mlflow.register_model(
            run_records["incremental-challenger"]["model_uri"],
            REGISTRY_NAME,
        ).version
    )
    client.set_registered_model_alias(REGISTRY_NAME, "candidate", challenger_version)
    try:
        champion_version = str(client.get_model_version_by_alias(REGISTRY_NAME, "champion").version)
    except MlflowException:
        champion_version = "uninitialized"

    return {
        "registered_model": REGISTRY_NAME,
        "initial_champion_version": initial_version,
        "candidate_version": challenger_version,
        "candidate_alias": f"{REGISTRY_NAME}@candidate",
        "champion_version_before_decision": champion_version,
        "runs": run_records,
    }


def _promote_registered_challenger(
    registration: dict[str, object],
) -> dict[str, object]:
    """Move the local MLflow champion alias after the policy has passed."""
    client = MlflowClient()
    version = str(registration["candidate_version"])
    client.set_registered_model_alias(REGISTRY_NAME, "champion", version)
    return {
        "action": "promote",
        "champion_version_after_decision": version,
        "previous_champion_version": registration["champion_version_before_decision"],
    }


def rollback_registered_champion(previous_version: str) -> None:
    """Restore the MLflow champion alias without retraining."""
    if not previous_version.strip():
        raise ValueError("previous_version cannot be empty")
    MlflowClient().set_registered_model_alias(
        REGISTRY_NAME,
        "champion",
        previous_version,
    )


def _sha256(path: Path) -> str:
    import hashlib

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


def run_retraining(
    train_path: Path = Path("data/raw/train_FD001.txt"),
    candidate_path: Path = Path("configs/xgboost-candidate-v1.json"),
    output_dir: Path = Path("reports/retraining"),
    model_dir: Path = Path("models"),
) -> dict[str, object]:
    """Run the fixed initial/challenger retraining simulation."""
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    labeled = add_rul_labels(load_fd001_file(train_path))
    manifests = build_engine_manifests(labeled["unit_id"].unique())
    initial_samples = generate_training_samples(labeled, manifests["initial_training"])
    development_samples = generate_training_samples(labeled, manifests["development"])
    validation_samples = generate_validation_samples(labeled, manifests["validation"]).samples

    initial_model_path = model_dir / "champion_initial_v1.joblib"
    initial_preprocessor_path = model_dir / "feature_preprocessor_initial_v1.joblib"
    challenger_model_path = model_dir / "challenger_incremental_v1.joblib"
    challenger_preprocessor_path = model_dir / "feature_preprocessor_challenger_v1.joblib"

    initial_metrics, initial_predictions, initial_batch_ms = _train_partition(
        labeled,
        initial_samples,
        validation_samples,
        model_path=initial_model_path,
        preprocessor_path=initial_preprocessor_path,
        candidate_params=candidate["best_params"],
    )
    challenger_metrics, challenger_predictions, challenger_batch_ms = _train_partition(
        labeled,
        development_samples,
        validation_samples,
        model_path=challenger_model_path,
        preprocessor_path=challenger_preprocessor_path,
        candidate_params=candidate["best_params"],
    )

    initial_model = joblib.load(initial_model_path)
    challenger_model = joblib.load(challenger_model_path)
    challenger_preprocessor = FeaturePreprocessor.load(challenger_preprocessor_path)
    validation_features = challenger_preprocessor.transform(
        build_feature_matrix(labeled, validation_samples)
    )
    latency_p95 = _p95_single_row_latency(challenger_model, validation_features)
    artifacts_registered = all(
        path.exists()
        for path in (
            initial_model_path,
            initial_preprocessor_path,
            challenger_model_path,
            challenger_preprocessor_path,
        )
    )
    tests_passed, test_output = _run_promotion_tests()
    registration = _register_retraining_models(
        candidate=candidate,
        train_path=train_path,
        initial_model=initial_model,
        challenger_model=challenger_model,
        initial_preprocessor_path=initial_preprocessor_path,
        challenger_preprocessor_path=challenger_preprocessor_path,
        initial_metrics=initial_metrics,
        challenger_metrics=challenger_metrics,
        initial_model_path=initial_model_path,
        challenger_model_path=challenger_model_path,
    )
    decision = evaluate_promotion(
        champion_rmse=initial_metrics["rmse"],
        champion_mae=initial_metrics["mae"],
        challenger_rmse=challenger_metrics["rmse"],
        challenger_mae=challenger_metrics["mae"],
        predictions=challenger_predictions.tolist(),
        expected_prediction_count=len(validation_samples),
        latency_p95_ms=latency_p95,
        artifacts_registered=artifacts_registered,
        tests_passed=tests_passed,
    )

    result: dict[str, object] = {
        "dataset": "NASA C-MAPSS FD001",
        "validation_engines": sorted(manifests["validation"]),
        "initial_training_engines": sorted(manifests["initial_training"]),
        "incremental_training_engines": sorted(manifests["incremental_training"]),
        "initial_training_samples": len(initial_samples),
        "challenger_training_samples": len(development_samples),
        "validation_samples": len(validation_samples),
        "champion": {
            "model": str(initial_model_path),
            "metrics": initial_metrics,
            "batch_inference_ms": initial_batch_ms,
            "prediction_bounds": [
                float(initial_predictions.min()),
                float(initial_predictions.max()),
            ],
        },
        "challenger": {
            "model": str(challenger_model_path),
            "metrics": challenger_metrics,
            "batch_inference_ms": challenger_batch_ms,
            "single_row_p95_ms": latency_p95,
            "prediction_bounds": [
                float(challenger_predictions.min()),
                float(challenger_predictions.max()),
            ],
        },
        "promotion": {
            "promote": decision.promote,
            "reasons": list(decision.reasons),
            "passed_gates": list(decision.passed_gates),
        },
        "verification": {
            "tests_passed": tests_passed,
            "test_command": "python -m pytest tests/unit -q --tb=short",
            "test_output_tail": test_output,
        },
        "mlflow": registration,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "champion-challenger.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "transition-audit.json").write_text(
        json.dumps(
            {
                "event": "promotion_decision",
                "decision": "promote" if decision.promote else "retain_champion",
                "reasons": list(decision.reasons),
                "rollback_backup": str(initial_model_path),
                "mlflow": registration,
                "promotion_action": (
                    _promote_registered_challenger(registration)
                    if decision.promote
                    else {"action": "retain_champion"}
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def retraining_triggered(
    *,
    consecutive_feature_psi: int,
    labeled_rmse: float | None,
    champion_validation_rmse: float,
) -> bool:
    """Return whether the fixed monitoring evidence requires retraining."""
    if consecutive_feature_psi < 0:
        raise ValueError("consecutive_feature_psi cannot be negative")
    if champion_validation_rmse <= 0:
        raise ValueError("champion_validation_rmse must be positive")
    if labeled_rmse is not None and labeled_rmse < 0:
        raise ValueError("labeled_rmse cannot be negative")

    drift_trigger = consecutive_feature_psi >= 2
    performance_trigger = (
        labeled_rmse is not None
        and labeled_rmse > RMSE_TRIGGER_MULTIPLIER * champion_validation_rmse
    )
    return drift_trigger or performance_trigger


def evaluate_promotion(
    *,
    champion_rmse: float,
    champion_mae: float,
    challenger_rmse: float,
    challenger_mae: float,
    predictions: list[float],
    expected_prediction_count: int,
    latency_p95_ms: float,
    artifacts_registered: bool,
    tests_passed: bool,
) -> PromotionDecision:
    """Apply every fixed promotion gate and return an auditable decision."""
    values = {
        "champion_rmse": champion_rmse,
        "champion_mae": champion_mae,
        "challenger_rmse": challenger_rmse,
        "challenger_mae": challenger_mae,
        "latency_p95_ms": latency_p95_ms,
    }
    if any(value < 0 for value in values.values()):
        raise ValueError("metrics and latency cannot be negative")
    if expected_prediction_count < 1:
        raise ValueError("expected_prediction_count must be positive")

    passed: list[str] = []
    rejected: list[str] = []
    improvement = (champion_rmse - challenger_rmse) / champion_rmse if champion_rmse else 0.0
    if improvement >= MIN_RMSE_IMPROVEMENT:
        passed.append("rmse_improvement")
    else:
        rejected.append("challenger RMSE improvement is below 3%")

    if challenger_mae <= champion_mae:
        passed.append("mae_not_worse")
    else:
        rejected.append("challenger MAE is worse than champion MAE")

    if len(predictions) == expected_prediction_count:
        passed.append("prediction_count")
    else:
        rejected.append("challenger did not produce all required predictions")

    bounded = all(RUL_MIN <= prediction <= RUL_MAX for prediction in predictions)
    if bounded:
        passed.append("prediction_bounds")
    else:
        rejected.append("challenger predictions are outside 0-125")

    if latency_p95_ms < MAX_VALIDATION_LATENCY_MS:
        passed.append("latency_p95")
    else:
        rejected.append("challenger p95 latency is not below 100 ms")

    if artifacts_registered:
        passed.append("artifact_registration")
    else:
        rejected.append("data, code, feature, metric, or model lineage is incomplete")

    if tests_passed:
        passed.append("tests")
    else:
        rejected.append("test suite did not pass")

    return PromotionDecision(not rejected, tuple(rejected), tuple(passed))
