"""Measure the deployed EngineGuard API with deterministic synthetic input."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def build_payload() -> dict[str, Any]:
    """Build the smallest valid 20-cycle prediction request."""
    observations = []
    for cycle in range(1, 21):
        observation: dict[str, Any] = {
            "cycle": cycle,
            "op_setting_1": 0.0,
            "op_setting_2": 0.0,
            "op_setting_3": 0.0,
        }
        observation.update({f"sensor_{index}": 0.0 for index in range(1, 22)})
        observations.append(observation)
    return {"unit_id": "load-test-engine", "observations": observations}


def post_prediction(url: str, payload: bytes) -> tuple[int, float, str]:
    """Send one request and return status, elapsed milliseconds, and response text."""
    request = Request(
        f"{url.rstrip('/')}/v1/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=60) as response:
            body = response.read().decode("utf-8")
            return response.status, (time.perf_counter() - started) * 1000, body
    except HTTPError as error:
        return error.code, (time.perf_counter() - started) * 1000, error.read().decode("utf-8")
    except URLError as error:
        return 0, (time.perf_counter() - started) * 1000, str(error.reason)


def percentile(values: list[float], fraction: float) -> float:
    """Return a linearly interpolated percentile."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + weight * (ordered[upper] - ordered[lower])


def measure(base_url: str, request_count: int, warmup_count: int) -> dict[str, Any]:
    """Measure first-request, warm-up, and sequential-request behavior."""
    payload = json.dumps(build_payload(), separators=(",", ":")).encode("utf-8")
    first_status, first_latency, first_body = post_prediction(base_url, payload)

    warmup = [post_prediction(base_url, payload) for _ in range(warmup_count)]
    samples = [post_prediction(base_url, payload) for _ in range(request_count)]
    statuses = [status for status, _, _ in samples]
    latencies = [latency for _, latency, _ in samples]
    successful = sum(status == 200 for status in statuses)

    return {
        "endpoint": base_url.rstrip("/") + "/v1/predict",
        "request_count": request_count,
        "warmup_count": warmup_count,
        "first_request": {
            "status": first_status,
            "latency_ms": round(first_latency, 3),
            "response_preview": first_body[:200],
        },
        "warmup_statuses": [status for status, _, _ in warmup],
        "sequential": {
            "successful_requests": successful,
            "error_count": request_count - successful,
            "error_rate": round((request_count - successful) / request_count, 6),
            "min_ms": round(min(latencies), 3),
            "mean_ms": round(statistics.mean(latencies), 3),
            "p50_ms": round(percentile(latencies, 0.50), 3),
            "p95_ms": round(percentile(latencies, 0.95), 3),
            "p99_ms": round(percentile(latencies, 0.99), 3),
            "max_ms": round(max(latencies), 3),
            "statuses": sorted(set(statuses)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="https://nasa-c-mapss.vercel.app")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("reports/deployment-load-test.json"))
    args = parser.parse_args()

    result = measure(args.url, args.requests, args.warmup)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
