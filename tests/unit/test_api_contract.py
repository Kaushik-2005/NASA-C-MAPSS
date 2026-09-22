import pytest
from fastapi.testclient import TestClient

from api.index import app as vercel_app
from src.api.app import app
from src.api.service import ModelService


def _observation(cycle: int) -> dict[str, float | int]:
    observation: dict[str, float | int] = {
        "cycle": cycle,
        "op_setting_1": 0.0,
        "op_setting_2": 0.0,
        "op_setting_3": 100.0,
    }
    observation.update({f"sensor_{index}": float(index) for index in range(1, 22)})
    return observation


def _history(unit_id: str = "engine-001") -> dict[str, object]:
    return {
        "unit_id": unit_id,
        "observations": [_observation(cycle) for cycle in range(1, 21)],
    }


def test_health_and_readiness_contracts() -> None:
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "healthy"}
        ready_response = client.get("/ready")

    assert ready_response.status_code == 200
    assert ready_response.json()["status"] == "ready"


def test_vercel_entrypoint_exports_same_fastapi_app() -> None:
    assert vercel_app is app


def test_prediction_response_contract() -> None:
    with TestClient(app) as client:
        response = client.post("/v1/predict", json=_history())

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "unit_id",
        "predicted_rul",
        "risk_level",
        "model_version",
        "feature_schema_version",
        "request_id",
        "predicted_at",
    }
    assert 0 <= body["predicted_rul"] <= 125
    assert body["feature_schema_version"] == "v1"


def test_batch_endpoint_and_model_info() -> None:
    payload = {"histories": [_history("engine-001"), _history("engine-002")]}

    with TestClient(app) as client:
        batch_response = client.post("/v1/predict/batch", json=payload)
        model_response = client.get("/model-info")

    assert batch_response.status_code == 200
    assert len(batch_response.json()["predictions"]) == 2
    assert model_response.status_code == 200
    assert model_response.json()["target"] == "target_rul"


def test_invalid_history_is_rejected() -> None:
    payload = _history()
    payload["observations"] = [_observation(1) for _ in range(20)]

    with TestClient(app) as client:
        response = client.post("/v1/predict", json=payload)

    assert response.status_code == 422
    assert response.json()["error_code"] == "VALIDATION_ERROR"
    assert response.json()["request_id"]


def test_request_size_limit_returns_typed_error() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/v1/predict",
            content=b"{}",
            headers={"content-length": "2000001"},
        )

    assert response.status_code == 413
    assert response.json()["error_code"] == "REQUEST_TOO_LARGE"


def test_optional_api_key_protects_prediction_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENGINEGUARD_API_KEY", "test-secret")
    with TestClient(app) as client:
        unauthorized = client.post("/v1/predict", json=_history())
        authorized = client.post(
            "/v1/predict",
            json=_history(),
            headers={"X-API-Key": "test-secret"},
        )

    assert unauthorized.status_code == 401
    assert unauthorized.json()["error_code"] == "AUTHENTICATION_REQUIRED"
    assert authorized.status_code == 200


def test_online_prediction_matches_shared_offline_service() -> None:
    payload = _history("engine-parity")
    offline_service = ModelService()
    expected = offline_service.predict(
        payload["unit_id"],
        payload["observations"],
    )

    with TestClient(app) as client:
        response = client.post("/v1/predict", json=payload)

    assert response.status_code == 200
    assert response.json()["predicted_rul"] == pytest.approx(expected)


def test_openapi_contains_all_required_endpoints() -> None:
    with TestClient(app) as client:
        schema = client.get("/openapi.json").json()

    assert set(schema["paths"]) >= {
        "/v1/predict",
        "/v1/predict/batch",
        "/health",
        "/ready",
        "/model-info",
    }


def test_batch_endpoint_rejects_more_than_100_histories() -> None:
    payload = {"histories": [_history(f"engine-{index}") for index in range(101)]}

    with TestClient(app) as client:
        response = client.post("/v1/predict/batch", json=payload)

    assert response.status_code == 422
