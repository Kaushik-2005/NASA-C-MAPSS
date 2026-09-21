"""Educational binary logistic regression trained with batch GD."""

from dataclasses import dataclass

import numpy as np


@dataclass
class LogisticRegressionGD:
    """Binary logistic regression using sigmoid cross-entropy loss."""

    learning_rate: float = 0.1
    epochs: int = 1_000
    l2: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionGD":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim != 2 or y.shape[0] != X.shape[0] or not np.all(np.isin(y, [0.0, 1.0])):
            raise ValueError("X must be 2-D and y must be binary with one value per row")
        Xb = np.column_stack((np.ones(X.shape[0]), X))
        self.weights_ = np.zeros(Xb.shape[1], dtype=float)
        self.loss_history_: list[float] = []
        for _ in range(self.epochs):
            probabilities = self._sigmoid(Xb @ self.weights_)
            clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
            loss = -np.mean(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped))
            self.loss_history_.append(float(loss))
            gradient = (Xb.T @ (probabilities - y)) / Xb.shape[0]
            gradient[1:] += self.l2 * self.weights_[1:]
            self.weights_ -= self.learning_rate * gradient
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "weights_"):
            raise RuntimeError("fit must be called before predict")
        X = np.asarray(X, dtype=float)
        scores = np.column_stack((np.ones(X.shape[0]), X)) @ self.weights_
        positive = self._sigmoid(scores)
        return np.column_stack((1.0 - positive, positive))

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(values, -500, 500)))
