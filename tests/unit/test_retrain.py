import pytest

from src.training import retrain
from src.training.retrain import (
    evaluate_promotion,
    retraining_triggered,
    rollback_model,
    rollback_registered_champion,
)


def test_retraining_trigger_requires_two_drift_batches_or_rmse_degradation() -> None:
    assert retraining_triggered(
        consecutive_feature_psi=2,
        labeled_rmse=None,
        champion_validation_rmse=10.0,
    )
    assert retraining_triggered(
        consecutive_feature_psi=0,
        labeled_rmse=12.1,
        champion_validation_rmse=10.0,
    )
    assert not retraining_triggered(
        consecutive_feature_psi=1,
        labeled_rmse=12.0,
        champion_validation_rmse=10.0,
    )


def test_promotion_requires_every_gate() -> None:
    decision = evaluate_promotion(
        champion_rmse=20.0,
        champion_mae=12.0,
        challenger_rmse=19.0,
        challenger_mae=11.5,
        predictions=[1.0, 60.0, 125.0],
        expected_prediction_count=3,
        latency_p95_ms=80.0,
        artifacts_registered=True,
        tests_passed=True,
    )
    assert decision.promote is True
    assert decision.reasons == ()


def test_failed_gate_prevents_promotion_and_records_reasons() -> None:
    decision = evaluate_promotion(
        champion_rmse=20.0,
        champion_mae=12.0,
        challenger_rmse=19.9,
        challenger_mae=12.1,
        predictions=[-1.0, 126.0],
        expected_prediction_count=3,
        latency_p95_ms=100.0,
        artifacts_registered=False,
        tests_passed=False,
    )
    assert decision.promote is False
    assert len(decision.reasons) == 7


def test_retraining_trigger_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        retraining_triggered(
            consecutive_feature_psi=-1,
            labeled_rmse=None,
            champion_validation_rmse=10.0,
        )


def test_rollback_restores_backup_without_retraining(tmp_path) -> None:
    active = tmp_path / "active.joblib"
    backup = tmp_path / "champion-backup.joblib"
    active.write_bytes(b"challenger")
    backup.write_bytes(b"champion")

    rollback_model(active, backup)

    assert active.read_bytes() == b"champion"


def test_registered_rollback_restores_alias_without_retraining(monkeypatch) -> None:
    calls: list[tuple[str, str, str]] = []

    class FakeClient:
        def set_registered_model_alias(
            self, registered_model: str, alias: str, version: str
        ) -> None:
            calls.append((registered_model, alias, version))

    monkeypatch.setattr(retrain, "MlflowClient", FakeClient)

    rollback_registered_champion("2")

    assert calls == [("EngineGuardRUL", "champion", "2")]
