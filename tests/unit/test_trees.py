import numpy as np

from src.training.trees import (
    CV_SPLITS,
    RANDOM_SEED,
    XGBOOST_SEARCH_ITERATIONS,
    build_grouped_cv,
    build_random_forest,
    build_xgboost,
    build_xgboost_search,
    validate_group_labels,
)


def test_grouped_cv_keeps_engines_disjoint() -> None:
    groups = np.repeat(np.arange(1, 11), 3)
    splits = list(build_grouped_cv().split(np.zeros((len(groups), 1)), groups=groups))

    assert len(splits) == CV_SPLITS

    for train_indices, validation_indices in splits:
        assert set(groups[train_indices]).isdisjoint(groups[validation_indices])


def test_xgboost_search_has_fixed_contract() -> None:
    search = build_xgboost_search()

    assert search.n_iter == XGBOOST_SEARCH_ITERATIONS
    assert search.random_state == RANDOM_SEED
    assert search.scoring == "neg_root_mean_squared_error"
    assert search.cv.n_splits == CV_SPLITS


def test_tree_builders_are_seeded() -> None:
    forest = build_random_forest()
    booster = build_xgboost()

    assert forest.random_state == RANDOM_SEED
    assert booster.random_state == RANDOM_SEED
    assert forest.n_jobs == 1
    assert booster.n_jobs == 1


def test_group_labels_require_five_engines() -> None:
    validate_group_labels(np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5]), 10)


def test_group_labels_reject_wrong_length() -> None:
    try:
        validate_group_labels(np.array([1, 2, 3, 4, 5]), 6)
    except ValueError as error:
        assert "one value per training sample" in str(error)
    else:
        raise AssertionError("Expected malformed groups to be rejected")
