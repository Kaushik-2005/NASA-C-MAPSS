"""Verify the two-consecutive-batch service-latency alert rule."""

from __future__ import annotations

import json
from pathlib import Path

from src.monitoring.drift import evaluate_monitoring_alerts


def main() -> None:
    source = Path("reports/monitoring/alert-log.json")
    output = Path("reports/monitoring/latency-alert-scenario.json")
    payload = json.loads(source.read_text(encoding="utf-8"))
    simulated_p95 = {6: 150.0, 7: 160.0}
    previous_breaches = 0

    for batch in payload["batches"]:
        if not batch["valid"]:
            previous_breaches = 0
            batch["simulated_service_p95_latency_ms"] = None
            batch["simulated_alerts"] = []
            continue
        service_p95 = simulated_p95.get(batch["batch"])
        alerts, previous_breaches = evaluate_monitoring_alerts(
            batch_rmse=batch.get("performance_rmse"),
            frozen_validation_rmse=16.427882310971146,
            service_p95_ms=service_p95,
            previous_latency_breaches=previous_breaches,
        )
        batch["simulated_service_p95_latency_ms"] = service_p95
        batch["simulated_alerts"] = alerts

    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    latency_alerts = [
        (batch["batch"], batch["simulated_alerts"])
        for batch in payload["batches"]
        if any(alert["type"] == "service_latency" for alert in batch["simulated_alerts"])
    ]
    print(f"service_latency_alerts={latency_alerts}")
    if [batch for batch, _ in latency_alerts] != [7]:
        raise SystemExit("Expected one service-latency alert on batch 7")


if __name__ == "__main__":
    main()
