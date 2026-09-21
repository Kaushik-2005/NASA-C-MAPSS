import numpy as np

from src.foundations.linear_regression import LinearRegressionGD
from src.foundations.logistic_regression import LogisticRegressionGD


def test_linear_regression_learns_rul_like_continuous_target() -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = 100.0 - 4.0 * X[:, 0]
    model = LinearRegressionGD(learning_rate=0.05, epochs=2_000).fit(X, y)
    assert np.max(np.abs(model.predict(X) - y)) < 0.1
    assert model.loss_history_[-1] < model.loss_history_[0]


def test_logistic_regression_learns_failure_within_30_boundary() -> None:
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1])
    model = LogisticRegressionGD(learning_rate=0.5, epochs=2_000).fit(X, y)
    assert np.array_equal(model.predict(X), y)
    assert model.loss_history_[-1] < model.loss_history_[0]


def test_regularization_does_not_penalize_intercept() -> None:
    X = np.zeros((3, 1))
    y = np.ones(3)
    model = LinearRegressionGD(learning_rate=0.1, epochs=100, l2=10.0).fit(X, y)
    assert model.weights_[0] > 0.9
