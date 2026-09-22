"""Deterministic feature-drift calculations for Module 14."""

from __future__ import annotations

from typing import Literal

import numpy as np

PsiSeverity = Literal["normal", "warning", "critical"]


def evaluate_monitoring_alerts(
    *,
    batch_rmse: float | None,
    frozen_validation_rmse: float,
    service_p95_ms: float | None,
    previous_latency_breaches: int,
    performance_multiplier: float = 1.20,
    latency_warning_ms: float = 100.0,
    latency_consecutive_batches: int = 2,
) -> tuple[list[dict[str, float | str]], int]:
    """Evaluate fixed performance and consecutive service-latency alert rules."""
    if frozen_validation_rmse <= 0 or performance_multiplier <= 0:
        raise ValueError("RMSE baseline and multiplier must be positive")
    if latency_consecutive_batches < 1:
        raise ValueError("latency_consecutive_batches must be positive")

    alerts: list[dict[str, float | str]] = []
    if batch_rmse is not None:
        performance_threshold = frozen_validation_rmse * performance_multiplier
        if batch_rmse > performance_threshold:
            alerts.append(
                {
                    "type": "performance_degradation",
                    "severity": "warning",
                    "batch_rmse": batch_rmse,
                    "threshold_rmse": performance_threshold,
                }
            )

    if service_p95_ms is not None and service_p95_ms > latency_warning_ms:
        latency_breaches = previous_latency_breaches + 1
    else:
        latency_breaches = 0
    if latency_breaches >= latency_consecutive_batches:
        alerts.append(
            {
                "type": "service_latency",
                "severity": "warning",
                "p95_latency_ms": service_p95_ms or 0.0,
                "threshold_ms": latency_warning_ms,
                "consecutive_breaches": float(latency_breaches),
            }
        )
    return alerts, latency_breaches


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    """Calculate PSI using reference quantile bins and stable proportions."""
    reference_values = np.asarray(reference, dtype=float)
    current_values = np.asarray(current, dtype=float)
    if reference_values.ndim != 1 or current_values.ndim != 1:
        raise ValueError("PSI inputs must be one-dimensional")
    if reference_values.size == 0 or current_values.size == 0:
        raise ValueError("PSI inputs cannot be empty")
    if not np.isfinite(reference_values).all() or not np.isfinite(current_values).all():
        raise ValueError("PSI inputs must be finite")
    if bins < 2:
        raise ValueError("PSI requires at least two bins")

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(reference_values, quantiles))
    if edges.size < 2:
        return 0.0 if np.isclose(reference_values.mean(), current_values.mean()) else float("inf")
    edges[0] = -np.inf
    edges[-1] = np.inf
    reference_counts, _ = np.histogram(reference_values, bins=edges)
    current_counts, _ = np.histogram(current_values, bins=edges)
    epsilon = 1e-6
    reference_share = np.maximum(reference_counts / reference_values.size, epsilon)
    current_share = np.maximum(current_counts / current_values.size, epsilon)
    return float(
        np.sum((current_share - reference_share) * np.log(current_share / reference_share))
    )


def psi_severity(value: float) -> PsiSeverity:
    """Map PSI to the fixed normal, warning, and critical thresholds."""
    if value > 0.30:
        return "critical"
    if value > 0.20:
        return "warning"
    return "normal"


def require_columns(columns: set[str], required: set[str]) -> None:
    """Reject a batch that does not satisfy the required feature schema."""
    missing = sorted(required - columns)
    if missing:
        raise ValueError(f"Monitoring batch is missing required fields: {missing}")
