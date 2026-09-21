"""Educational batch-gradient linear regression."""

from dataclasses import dataclass

import numpy as np


@dataclass
class LinearRegressionGD:
    """Linear regression trained with mean squared error and batch GD."""

    learning_rate: float = 0.01
    epochs: int = 1_000
    l2: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionGD":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim != 2 or y.shape[0] != X.shape[0]:
            raise ValueError("X must be 2-D and y must have one value per row")
        Xb = np.column_stack((np.ones(X.shape[0]), X))
        self.weights_ = np.zeros(Xb.shape[1], dtype=float)
        self.loss_history_: list[float] = []
        for _ in range(self.epochs):
            residual = Xb @ self.weights_ - y
            self.loss_history_.append(float(np.mean(residual**2)))
            gradient = (2.0 / Xb.shape[0]) * (Xb.T @ residual)
            gradient[1:] += 2.0 * self.l2 * self.weights_[1:]
            self.weights_ -= self.learning_rate * gradient
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "weights_"):
            raise RuntimeError("fit must be called before predict")
        X = np.asarray(X, dtype=float)
        return np.column_stack((np.ones(X.shape[0]), X)) @ self.weights_
