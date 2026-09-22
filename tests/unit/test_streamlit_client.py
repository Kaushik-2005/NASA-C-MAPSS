from __future__ import annotations

import json
from typing import Self

import pytest

from streamlit_app.api_client import ApiClientError, call_api


def test_call_api_decodes_json(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"status": "ready"}).encode()

    monkeypatch.setattr("streamlit_app.api_client.urlopen", lambda *_args, **_kwargs: Response())

    assert call_api("https://example.test", "/ready") == {"status": "ready"}


def test_call_api_surfaces_network_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise TimeoutError("timed out")

    monkeypatch.setattr("streamlit_app.api_client.urlopen", fail)

    with pytest.raises(ApiClientError, match="Could not reach"):
        call_api("https://example.test", "/ready")
