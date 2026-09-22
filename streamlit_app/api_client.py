"""Small HTTP client for the deployed EngineGuard API."""

from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass
class ApiClientError(RuntimeError):
    """Actionable error returned by the API client."""

    message: str
    status_code: int | None = None

    def __str__(self) -> str:
        if self.status_code is None:
            return self.message
        return f"{self.message} (HTTP {self.status_code})"


def call_api(
    base_url: str,
    endpoint: str,
    payload: dict[str, object] | None = None,
    *,
    timeout_seconds: float = 15.0,
) -> dict[str, object]:
    """Call a JSON EngineGuard endpoint without retrying invalid requests."""
    url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        method="POST" if body is not None else "GET",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
            result = json.loads(raw)
    except HTTPError as error:
        try:
            detail = json.loads(error.read().decode("utf-8"))
            message = str(detail.get("message", detail.get("detail", "API request failed")))
        except (json.JSONDecodeError, UnicodeDecodeError):
            message = "API request failed"
        raise ApiClientError(message, error.code) from error
    except (URLError, TimeoutError, OSError) as error:
        raise ApiClientError(f"Could not reach EngineGuard API: {error}") from error
    except json.JSONDecodeError as error:
        raise ApiClientError("EngineGuard API returned invalid JSON") from error

    if not isinstance(result, dict):
        raise ApiClientError("EngineGuard API returned an unexpected response")
    return result
