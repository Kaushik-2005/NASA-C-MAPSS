"""Request-safety helpers for the public API boundary."""

from __future__ import annotations

import hmac
import os
from collections.abc import Iterable
from typing import Any

MAX_REQUEST_BYTES = 2_000_000
REQUEST_TIMEOUT_SECONDS = 10.0
API_KEY_ENV_VAR = "ENGINEGUARD_API_KEY"


def validation_details(errors: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    """Return JSON-safe validation details without exposing request payloads."""
    return [
        {
            "location": ".".join(str(part) for part in error.get("loc", ())),
            "message": str(error.get("msg", "invalid request")),
            "type": str(error.get("type", "validation_error")),
        }
        for error in errors
    ]


def configured_api_key() -> str | None:
    """Return the optional API key used by non-public deployments."""
    value = os.getenv(API_KEY_ENV_VAR)
    return value if value else None


def api_key_is_valid(provided_key: str | None) -> bool:
    """Validate an API key without enabling authentication by default."""
    expected_key = configured_api_key()
    if expected_key is None or provided_key is None:
        return expected_key is None
    return hmac.compare_digest(provided_key, expected_key)
