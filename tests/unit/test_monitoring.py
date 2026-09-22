from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.ingest import load_fd001_file
from src.data.labels import add_rul_labels
from src.data.samples import generate_training_samples
from src.data.splits import build_engine_manifests
from src.monitoring.drift import (
    evaluate_monitoring_alerts,
    population_stability_index,
    psi_severity,
    require_columns,
)
from src.monitoring.pipeline import generate_monitoring_batches, write_evidently_drift_report


def test_psi_is_zero_for_identical_distributions() -> None:
    values = np.arange(100, dtype=float)
    assert population_stability_index(values, values) == pytest.approx(0.0)


def test_psi_detects_shift_and_uses_fixed_thresholds() -> None:
    reference = np.zeros(100)
    shifted = np.ones(100)
    assert population_stability_index(reference, shifted) > 0.30
    assert psi_severity(0.10) == "normal"
    assert psi_severity(0.21) == "warning"
    assert psi_severity(0.31) == "critical"


def test_monitoring_schema_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing required"):
        require_columns({"sensor_2"}, {"sensor_2", "sensor_11"})


def test_monitoring_alerts_require_performance_and_two_latency_breaches() -> None:
    alerts, breaches = evaluate_monitoring_alerts(
        batch_rmse=21.0,
        frozen_validation_rmse=16.0,
        service_p95_ms=150.0,
        previous_latency_breaches=0,
    )
    assert {alert["type"] for alert in alerts} == {"performance_degradation"}
    assert breaches == 1

    alerts, breaches = evaluate_monitoring_alerts(
        batch_rmse=10.0,
        frozen_validation_rmse=16.0,
        service_p95_ms=150.0,
        previous_latency_breaches=1,
    )
    assert {alert["type"] for alert in alerts} == {"service_latency"}
    assert breaches == 2


def test_fixed_monitoring_batch_sequence_quarantines_batch_nine() -> None:
    frame = add_rul_labels(load_fd001_file(Path("data/raw/train_FD001.txt")))
    manifests = build_engine_manifests(frame["unit_id"].unique())
    validation = frame[frame["unit_id"].isin(manifests["validation"])]
    samples = generate_training_samples(validation, manifests["validation"])
    sensor_std = frame[frame["unit_id"].isin(manifests["development"])]
    sensor_std = sensor_std[[f"sensor_{index}" for index in range(1, 22)]].std(ddof=0)

    batches = generate_monitoring_batches(validation, samples, sensor_std, batch_size=2)

    assert len(batches) == 10
    assert all(batch.valid for batch in batches[:8])
    assert batches[8].valid is False
    assert "sensor_5" in (batches[8].error or "")
    assert batches[9].valid is True


def test_evidently_report_writes_html_and_json(tmp_path: Path) -> None:
    reference = pd.DataFrame({"sensor_2__current": np.arange(30, dtype=float)})
    current = pd.DataFrame({"sensor_2__current": np.arange(30, dtype=float) + 2.0})

    paths = write_evidently_drift_report(reference, current, tmp_path, 1)

    assert Path(paths["html"]).exists()
    assert Path(paths["json"]).exists()
