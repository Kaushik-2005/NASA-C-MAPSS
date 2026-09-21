import pandas as pd
import pytest

from src.features.build_features import (
    FEATURE_COLUMNS,
    FEATURE_SCHEMA_VERSION,
    RETAINED_COLUMNS,
    build_features,
    build_feature_matrix,
    feature_metadata,
    write_feature_metadata,
)


def make_history(cycles: int) -> pd.DataFrame:
    frame =  pd.DataFrame(
        {
            "unit_id": [1] * cycles,
            "cycle": list(range(1, cycles + 1)),
        }
    )

    for column in RETAINED_COLUMNS:
        frame[column] = range(cycles)

    return frame 


def test_features_reject_short_history() -> None:
    history = make_history(cycles=19)

    with pytest.raises(ValueError, match="20 history cycles"):
        build_features(history)


def test_features_reject_duplicate_cycles() -> None:
    history = make_history(20)
    history.loc[19, "cycle"] = 19

    with pytest.raises(ValueError, match="duplicates"):
        build_features(history)


def test_features_return_current_values_and_history_count() -> None:
    history = make_history(20)

    features = build_features(history)

    assert len(features) == 1
    assert features["cycle__current"].iloc[0] == 20
    assert features["history_count"].iloc[0] == 20
    assert "sensor_1__current" not in features.columns
    assert "sensor_2__current" in features.columns


def test_features_include_rolling_statistics() -> None:
    history = make_history(20)

    features = build_features(history)

    assert "sensor_2__rolling_mean_5" in features.columns
    assert "sensor_2__rolling_std_20" in features.columns
    assert features["sensor_2__rolling_mean_5"].iloc[0] == 17


def test_features_include_delta_and_slope() -> None:
    history = make_history(20)

    features = build_features(history)

    assert features["sensor_2__delta_5"].iloc[0] == 5
    assert features["sensor_2__slope_20"].iloc[0] == pytest.approx(1.0)


def test_feature_columns_are_ordered_and_versioned() -> None:
    history = make_history(20)

    features = build_features(history)

    assert tuple(features.columns) == FEATURE_COLUMNS
    assert FEATURE_SCHEMA_VERSION == "v1"
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))


def test_equivalent_histories_have_identical_features() -> None:
    first_history = make_history(20)
    second_history = first_history.copy(deep=True)

    first_features = build_features(first_history)
    second_features = build_features(second_history)

    pd.testing.assert_frame_equal(first_features, second_features)


def test_longer_histories_are_accepted_and_history_count_is_capped() -> None:
    history = make_history(25)

    features = build_features(history)

    assert features["history_count"].iloc[0] == 20
    assert features["sensor_2__rolling_mean_5"].iloc[0] == 22


def test_features_from_prefix_do_not_use_future_rows() -> None:
    history = make_history(25)
    prefix = history[history["cycle"] <= 20].copy()
    altered_future = history.copy()
    altered_future.loc[altered_future["cycle"] > 20, RETAINED_COLUMNS] = 999999

    prefix_features = build_features(prefix)
    altered_future_prefix_features = build_features(
        altered_future[altered_future["cycle"] <= 20]
    )

    pd.testing.assert_frame_equal(
        prefix_features,
        altered_future_prefix_features,
    )


def test_serialized_feature_metadata_preserves_order(tmp_path) -> None:
    metadata_path = tmp_path / "feature_schema_v1.json"

    write_feature_metadata(metadata_path)
    metadata = __import__("json").loads(metadata_path.read_text())

    assert metadata == feature_metadata()
    assert tuple(metadata["feature_columns"]) == FEATURE_COLUMNS
    assert metadata["feature_schema_version"] == "v1"


def test_feature_matrix_aligns_rows_to_samples() -> None:
    trajectories = make_history(25)
    samples = pd.DataFrame(
        {
            "unit_id": [1, 1],
            "cycle": [20, 25],
            "target_rul": [5, 0],
        }
    )

    matrix = build_feature_matrix(trajectories, samples)

    assert matrix.shape == (2, len(FEATURE_COLUMNS))
    assert matrix["cycle__current"].tolist() == [20, 25]
    assert "target_rul" not in matrix.columns


def test_feature_matrix_does_not_use_rows_after_sample_cycle() -> None:
    trajectories = make_history(25)
    altered = trajectories.copy()
    altered.loc[altered["cycle"] > 20, RETAINED_COLUMNS] = 999999
    samples = pd.DataFrame({"unit_id": [1], "cycle": [20]})

    original_matrix = build_feature_matrix(trajectories, samples)
    altered_matrix = build_feature_matrix(altered, samples)

    pd.testing.assert_frame_equal(original_matrix, altered_matrix)
