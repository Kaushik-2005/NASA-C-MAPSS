from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.evaluation.final_evaluation import (
    _json_default,
    _load_candidate,
    _shap_artifacts,
    _write_model_card,
    _write_report,
    nasa_asymmetric_score,
)


def test_nasa_score_is_zero_for_perfect_predictions() -> None:
    values = np.array([0.0, 30.0, 125.0])

    assert nasa_asymmetric_score(values, values) == pytest.approx(0.0)


def test_nasa_score_penalizes_underprediction_asymmetrically() -> None:
    y_true = np.array([10.0])
    underprediction = np.array([0.0])
    overprediction = np.array([20.0])

    under_score = nasa_asymmetric_score(y_true, underprediction)
    over_score = nasa_asymmetric_score(y_true, overprediction)

    assert under_score == pytest.approx(np.exp(10 / 13) - 1)
    assert over_score == pytest.approx(np.exp(10 / 10) - 1)


def test_nasa_score_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        nasa_asymmetric_score(np.array([1.0]), np.array([1.0, 2.0]))


def test_final_evaluation_helpers_write_reports_and_shap_artifacts(tmp_path) -> None:
    candidate = _load_candidate(Path("configs/xgboost-candidate-v1.json"))
    preprocessor = joblib.load("models/feature_preprocessor_v1.joblib")
    model = joblib.load("models/xgboost_candidate_v1.joblib")
    feature_names = tuple(preprocessor.selected_feature_names_ or ())
    samples = pd.DataFrame(
        {
            "unit_id": [1, 2, 3],
            "cycle": [20, 30, 40],
        }
    )

    explainability_dir = tmp_path / "explainability"
    explainability_dir.mkdir()
    artifacts = _shap_artifacts(
        model,
        np.zeros((3, len(feature_names)), dtype=float),
        samples,
        feature_names,
        explainability_dir,
    )

    assert (explainability_dir / "shap-global-importance.csv").exists()
    assert artifacts["local_rows"] == [0, 1, 2]

    metrics = {
        "prediction_count": 3,
        "prediction_range": [0.0, 10.0],
        "regression": {"rmse": 1.0, "mae": 0.5, "r_squared": 0.9},
        "nasa_asymmetric_score": 2.0,
        "within_tolerance": {"within_10": 3, "within_20": 3, "within_40": 3},
        "error_bands": {"0-30": {"count": 3, "rmse": 1.0, "mae": 0.5}},
    }
    report_path = tmp_path / "final-evaluation.md"
    card_path = tmp_path / "model-card.md"
    _write_report(metrics, report_path)
    _write_model_card(card_path, metrics, candidate)
    assert "Final FD001 Evaluation" in report_path.read_text(encoding="utf-8")
    assert "EngineGuard Model Card" in card_path.read_text(encoding="utf-8")


def test_prepare_data_builds_development_and_final_holdout_matrices() -> None:
    from src.evaluation.final_evaluation import _prepare_data

    X_train, X_test, y_train, y_test, samples, preprocessor = _prepare_data(
        Path("data/raw/train_FD001.txt"),
        Path("data/raw/test_FD001.txt"),
        Path("data/raw/RUL_FD001.txt"),
    )

    assert X_train.shape[0] > 0
    assert X_test.shape == (100, X_train.shape[1])
    assert y_train.shape[0] == X_train.shape[0]
    assert y_test.shape == (100,)
    assert len(samples) == 100
    assert preprocessor.selected_feature_names_


def test_final_evaluation_rejects_invalid_candidate_and_empty_score(tmp_path: Path) -> None:
    invalid = tmp_path / "candidate.json"
    invalid.write_text('{"feature_schema_version": "v0"}', encoding="utf-8")
    with pytest.raises(ValueError, match="feature schema"):
        _load_candidate(invalid)
    with pytest.raises(FileNotFoundError, match="not found"):
        _load_candidate(tmp_path / "missing.json")
    with pytest.raises(ValueError, match="cannot be empty"):
        nasa_asymmetric_score(np.array([]), np.array([]))


def test_final_evaluation_json_default_handles_numpy_values() -> None:
    assert _json_default(np.int64(4)) == 4
    assert _json_default(np.float64(2.5)) == 2.5
    assert _json_default(np.array([1, 2])) == [1, 2]
    with pytest.raises(TypeError, match="Unsupported JSON value"):
        _json_default(object())
