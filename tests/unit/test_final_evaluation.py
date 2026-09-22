import numpy as np
import pytest

from src.evaluation.final_evaluation import nasa_asymmetric_score


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
