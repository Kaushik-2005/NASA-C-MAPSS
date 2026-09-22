from pathlib import Path

import numpy as np

from src.training.train import _git_revision, _prepare_training_data, _sha256


def test_sha256_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("EngineGuard", encoding="utf-8")

    assert _sha256(path) == _sha256(path)
    assert len(_sha256(path)) == 64


def test_git_revision_returns_a_traceable_value() -> None:
    revision = _git_revision()

    assert revision


def test_prepare_training_data_respects_fixed_partitions() -> None:
    X_train, X_validation, y_train, y_validation, preprocessor = _prepare_training_data(
        Path("data/raw/train_FD001.txt"),
        Path("models/feature_preprocessor_v1.joblib"),
    )

    assert X_train.shape[0] > X_validation.shape[0]
    assert X_train.shape[1] == X_validation.shape[1]
    assert len(y_train) == X_train.shape[0]
    assert len(y_validation) == X_validation.shape[0]
    assert np.isfinite(X_train).all()
    assert preprocessor.feature_names_ is not None
    assert preprocessor.selected_feature_names_ is not None
