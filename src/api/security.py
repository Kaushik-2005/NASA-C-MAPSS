"""Request-safety helpers for the public API boundary."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

MAX_REQUEST_BYTES = 2_000_000
REQUEST_TIMEOUT_SECONDS = 10.0


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
