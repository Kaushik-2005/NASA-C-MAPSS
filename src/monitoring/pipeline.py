"""Deterministic monitoring batches and drift reporting for FD001."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.api.service import risk_level
from src.config import PROJECT_CONFIG
from src.data.ingest import load_fd001_file
from src.data.labels import add_rul_labels
from src.data.samples import generate_training_samples
from src.data.splits import build_engine_manifests
from src.features.build_features import (
    FEATURE_COLUMNS,
    build_feature_matrix,
    build_features,
)
from src.features.preprocessing import FeaturePreprocessor
from src.monitoring.drift import (
    evaluate_monitoring_alerts,
    population_stability_index,
    psi_severity,
    require_columns,
)

RAW_SENSOR_COLUMNS = {f"sensor_{index}" for index in range(1, 22)}
RAW_MONITORING_COLUMNS = {
    "unit_id",
    "cycle",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
    *RAW_SENSOR_COLUMNS,
}
BATCH_COUNT = 10
BATCH_SIZE = 1000
REFERENCE_SIZE = 5000


@dataclass(frozen=True)
class MonitoringBatchResult:
    """Result of processing one deterministic monitoring batch."""

    batch_number: int
    scenario: str
    valid: bool
    features: pd.DataFrame | None
    targets: np.ndarray | None
    error: str | None


def write_evidently_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_dir: Path,
    batch_number: int,
) -> dict[str, str]:
    """Write Evidently HTML and JSON comparison reports for one valid batch."""
    from evidently import Report
    from evidently.presets import DataDriftPreset

    output_dir.mkdir(parents=True, exist_ok=True)
    report = Report(
        metrics=[DataDriftPreset()],
        metadata={"feature_schema_version": PROJECT_CONFIG.feature_schema_version},
        include_tests=True,
    )
    snapshot = report.run(current_data=current, reference_data=reference)
    html_path = output_dir / f"batch-{batch_number:02d}-drift.html"
    json_path = output_dir / f"batch-{batch_number:02d}-drift.json"
    snapshot.save_html(str(html_path))
    snapshot.save_json(str(json_path))
    return {"html": str(html_path), "json": str(json_path)}


def build_reference_feature_sample(
    train_path: Path = Path("data/raw/train_FD001.txt"),
    preprocessor_path: Path = Path("models/feature_preprocessor_v1.joblib"),
    *,
    sample_size: int = REFERENCE_SIZE,
    seed: int = PROJECT_CONFIG.random_seed,
) -> pd.DataFrame:
    """Build the fixed seed-42 reference sample from development features only."""
    labeled = add_rul_labels(load_fd001_file(train_path))
    manifests = build_engine_manifests(labeled["unit_id"].unique())
    development_samples = generate_training_samples(labeled, manifests["development"])
    raw_features = build_feature_matrix(labeled, development_samples)
    preprocessor = FeaturePreprocessor.load(preprocessor_path)
    transformed = preprocessor.transform(raw_features)
    if len(transformed) < sample_size:
        raise ValueError(f"Reference data contains {len(transformed)} rows; {sample_size} required")
    rng = np.random.default_rng(seed)
    positions = rng.choice(len(transformed), size=sample_size, replace=False)
    return transformed.iloc[np.sort(positions)].reset_index(drop=True)


def _shifted_history(
    history: pd.DataFrame,
    *,
    sensor_2_shift: float = 0.0,
    sensor_11_shift: float = 0.0,
    remove_sensor_5: bool = False,
) -> pd.DataFrame:
    shifted = history.copy()
    if remove_sensor_5:
        shifted = shifted.drop(columns=["sensor_5"])
    else:
        shifted["sensor_2"] = shifted["sensor_2"] + sensor_2_shift
        shifted["sensor_11"] = shifted["sensor_11"] + sensor_11_shift
    return shifted


def _feature_batch(
    trajectories: pd.DataFrame,
    samples: pd.DataFrame,
    positions: np.ndarray,
    *,
    sensor_2_shift: float = 0.0,
    sensor_11_shift: float = 0.0,
    remove_sensor_5: bool = False,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for position in positions:
        sample = samples.iloc[int(position)]
        history = trajectories[
            (trajectories["unit_id"] == sample["unit_id"])
            & (trajectories["cycle"] <= sample["cycle"])
        ].sort_values("cycle")
        shifted = _shifted_history(
            history,
            sensor_2_shift=sensor_2_shift,
            sensor_11_shift=sensor_11_shift,
            remove_sensor_5=remove_sensor_5,
        )
        rows.append(build_features(shifted))
    return pd.concat(rows, ignore_index=True).loc[:, FEATURE_COLUMNS]


def generate_monitoring_batches(
    validation_trajectories: pd.DataFrame,
    validation_samples: pd.DataFrame,
    development_sensor_std: pd.Series,
    *,
    batch_size: int = BATCH_SIZE,
    seed: int = PROJECT_CONFIG.random_seed,
) -> list[MonitoringBatchResult]:
    """Generate the ten fixed monitoring scenarios with seed-42 sampling."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    require_columns(set(validation_trajectories.columns), RAW_MONITORING_COLUMNS)
    rng = np.random.default_rng(seed)
    results: list[MonitoringBatchResult] = []
    for batch_number in range(1, BATCH_COUNT + 1):
        positions = rng.integers(0, len(validation_samples), size=batch_size)
        if 1 <= batch_number <= 5 or batch_number == 10:
            scenario = "normal"
            kwargs: dict[str, Any] = {}
        elif 6 <= batch_number <= 8:
            scenario = "sensor_shift"
            kwargs = {
                "sensor_2_shift": 2.0 * float(development_sensor_std["sensor_2"]),
                "sensor_11_shift": -1.5 * float(development_sensor_std["sensor_11"]),
            }
        else:
            scenario = "invalid_sensor_schema"
            kwargs = {"remove_sensor_5": True}

        try:
            if scenario == "invalid_sensor_schema":
                require_columns(
                    set(validation_trajectories.drop(columns=["sensor_5"]).columns),
                    RAW_MONITORING_COLUMNS,
                )
            features = _feature_batch(
                validation_trajectories, validation_samples, positions, **kwargs
            )
            targets = (
                validation_samples.iloc[positions]["target_rul"].to_numpy(dtype=float)
                if "target_rul" in validation_samples.columns
                else None
            )
            results.append(
                MonitoringBatchResult(batch_number, scenario, True, features, targets, None)
            )
        except ValueError as error:
            results.append(
                MonitoringBatchResult(batch_number, scenario, False, None, None, str(error))
            )
    return results


def run_monitoring(
    train_path: Path = Path("data/raw/train_FD001.txt"),
    preprocessor_path: Path = Path("models/feature_preprocessor_v1.joblib"),
    model_path: Path = Path("models/xgboost_candidate_v1.joblib"),
    output_dir: Path = Path("reports/monitoring"),
    *,
    seed: int = PROJECT_CONFIG.random_seed,
    service_p95_by_batch: dict[int, float] | None = None,
) -> dict[str, Any]:
    """Run the fixed monitoring simulation and write per-batch JSON reports."""
    labeled = add_rul_labels(load_fd001_file(train_path))
    manifests = build_engine_manifests(labeled["unit_id"].unique())
    development = labeled[labeled["unit_id"].isin(manifests["development"])]
    validation = labeled[labeled["unit_id"].isin(manifests["validation"])]
    validation_samples = generate_training_samples(validation, manifests["validation"])
    reference = build_reference_feature_sample(train_path, preprocessor_path, seed=seed)
    preprocessor = FeaturePreprocessor.load(preprocessor_path)
    model = joblib.load(model_path)
    sensor_std = development[list(RAW_SENSOR_COLUMNS)].std(ddof=0)
    batches = generate_monitoring_batches(validation, validation_samples, sensor_std, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, Any]] = []
    latency_breaches = 0

    for batch in batches:
        record: dict[str, Any] = {
            "batch": batch.batch_number,
            "scenario": batch.scenario,
            "schema_version": PROJECT_CONFIG.feature_schema_version,
            "model_version": "EngineGuardRUL@champion",
            "volume": BATCH_SIZE,
            "valid": batch.valid,
            "error": batch.error,
        }
        if batch.valid and batch.features is not None:
            transformed = preprocessor.transform(batch.features)
            drift = {
                column: population_stability_index(
                    reference[column].to_numpy(), transformed[column].to_numpy()
                )
                for column in reference.columns
            }
            predictions_started = time.perf_counter()
            predictions = np.clip(np.asarray(model.predict(transformed), dtype=float), 0, 125)
            latency_ms = (time.perf_counter() - predictions_started) * 1000
            risks = (
                pd.Series([risk_level(float(value)) for value in predictions])
                .value_counts()
                .to_dict()
            )
            batch_rmse = None
            if batch.targets is not None:
                batch_rmse = float(np.sqrt(np.mean((batch.targets - predictions) ** 2)))
            service_p95_ms = (
                None
                if service_p95_by_batch is None
                else service_p95_by_batch.get(batch.batch_number)
            )
            alerts, latency_breaches = evaluate_monitoring_alerts(
                batch_rmse=batch_rmse,
                frozen_validation_rmse=16.427882310971146,
                service_p95_ms=service_p95_ms,
                previous_latency_breaches=latency_breaches,
            )
            record.update(
                {
                    "psi": {name: round(value, 6) for name, value in drift.items()},
                    "max_psi": round(max(drift.values()), 6),
                    "max_psi_severity": psi_severity(max(drift.values())),
                    "prediction_latency_ms": round(latency_ms, 3),
                    "service_p95_latency_ms": service_p95_ms,
                    "performance_rmse": batch_rmse,
                    "alerts": alerts,
                    "prediction_range": [float(predictions.min()), float(predictions.max())],
                    "risk_distribution": {key: int(value) for key, value in risks.items()},
                    "evidently": write_evidently_drift_report(
                        reference, transformed, output_dir / "evidently", batch.batch_number
                    ),
                }
            )
        else:
            latency_breaches = 0
            record["alerts"] = []
        path = output_dir / f"batch-{batch.batch_number:02d}.json"
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        summary.append(record)

    result = {"seed": seed, "reference_rows": len(reference), "batches": summary}
    (output_dir / "alert-log.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    result = run_monitoring()
    for batch in result["batches"]:
        print(
            f"batch={batch['batch']} valid={batch['valid']} "
            f"severity={batch.get('max_psi_severity', 'quarantined')}"
        )
