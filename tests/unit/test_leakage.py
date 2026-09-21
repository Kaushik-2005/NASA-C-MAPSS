import pandas as pd
import pytest

from src.data.leakage import (
    assert_disjoint_partitions,
    assert_feature_columns_are_safe,
    assert_samples_belong_to_engines,
)


def test_disjoint_partitions_pass() -> None:
    assert_disjoint_partitions({"development": {1, 2}, "validation": {5}})


def test_overlapping_partitions_fail() -> None:
    with pytest.raises(AssertionError, match="overlap"):
        assert_disjoint_partitions({"development": {1}, "validation": {1}})


def test_label_columns_are_rejected_as_features() -> None:
    with pytest.raises(AssertionError, match="target_rul"):
        assert_feature_columns_are_safe(["sensor_2", "target_rul"])


def test_safe_feature_columns_pass() -> None:
    assert_feature_columns_are_safe(["cycle", "sensor_2"])


def test_samples_must_belong_to_manifest() -> None:
    samples = pd.DataFrame({"unit_id": [1, 3], "cycle": [20, 20]})

    with pytest.raises(AssertionError, match="unexpected engines"):
        assert_samples_belong_to_engines(samples, {1, 2})
