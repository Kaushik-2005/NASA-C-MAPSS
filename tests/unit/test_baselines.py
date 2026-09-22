import numpy as np
import pytest

from src.evaluation.metrics import classification_metrics, mae, regression_metrics, rmse
from src.training.baselines import (
    LogisticFailureClassifier,
    MedianRULRegressor,
    RidgeRULRegressor,
)


def test_median_regressor_predicts_training_median() -> None:
    X = np.zeros((4, 2))
    y = np.array([10.0, 20.0, 30.0, 40.0])

    model = MedianRULRegressor()
    model.fit(X, y)

    predictions = model.predict(np.ones((3, 2)))

    np.testing.assert_allclose(predictions, [25.0, 25.0, 25.0])


def test_median_regressor_requires_fit() -> None:
    model = MedianRULRegressor()

    with pytest.raises(RuntimeError):
        model.predict(np.zeros((2, 2)))


def test_regression_metrics() -> None:
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])

    assert mae(y_true, y_pred) == pytest.approx(7 / 3)
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(17 / 3))

    metrics = regression_metrics(y_true, y_pred)

    assert set(metrics) == {"rmse", "mae", "r_squared"}
    assert metrics["r_squared"] == pytest.approx(0.915)


def test_metrics_reject_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        mae(np.array([1.0, 2.0]), np.array([1.0]))


def test_ridge_regressor_learns_linear_relationship() -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([5.0, 7.0, 9.0, 11.0])

    model = RidgeRULRegressor(alpha=1e-8)
    model.fit(X, y)

    predictions = model.predict(np.array([[4.0], [5.0]]))

    np.testing.assert_allclose(predictions, [13.0, 15.0], atol=1e-4)


def test_ridge_regressor_rejects_invalid_alpha() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        RidgeRULRegressor(alpha=0.0)


def test_ridge_regressor_requires_matching_sample_counts() -> None:
    model = RidgeRULRegressor()

    with pytest.raises(ValueError, match="same number of samples"):
        model.fit(np.zeros((2, 1)), np.zeros(1))


def test_classification_metrics_include_threshold_and_confusion_matrix() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.6, 0.9])

    metrics = classification_metrics(y_true, y_score)

    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["pr_auc"] == pytest.approx(1.0)
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]


def test_classification_metrics_reject_non_binary_labels() -> None:
    with pytest.raises(ValueError, match="binary labels"):
        classification_metrics(np.array([0.0, 0.5, 1.0]), np.array([0.1, 0.5, 0.9]))


def test_logistic_classifier_supports_recall_oriented_threshold() -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([0, 0, 0, 1, 1, 1])

    model = LogisticFailureClassifier()
    model.fit(X, y)

    default_predictions = model.predict(X)
    recall_predictions = model.predict_with_threshold(X, threshold=0.3)

    assert default_predictions.shape == y.shape
    assert recall_predictions.shape == y.shape
    assert recall_predictions.sum() >= default_predictions.sum()


def test_logistic_classifier_rejects_single_class_target() -> None:
    model = LogisticFailureClassifier()

    with pytest.raises(ValueError, match="both binary classes"):
        model.fit(np.zeros((3, 1)), np.zeros(3))
