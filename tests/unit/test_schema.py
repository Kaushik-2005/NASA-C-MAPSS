import pytest
import pandas as pd

from src.data.schema import (
    EXPECTED_COLUMNS,
    validate_column_count,
    validate_column_names,
    validate_trajectory_data,
)


def test_expected_columns_length() -> None:
    assert len(EXPECTED_COLUMNS) == 26


def test_schema_starts_with_identity_columns() -> None:
    assert EXPECTED_COLUMNS[:2] == ("unit_id", "cycle")


def test_schema_ends_with_sensor_21() -> None:
    assert EXPECTED_COLUMNS[-1] == "sensor_21"


def test_valid_column_count_does_not_raise() -> None:
    validate_column_count(26)


def test_invalid_column_count_raises() -> None:
    with pytest.raises(ValueError, match="26.*25"):
        validate_column_count(25)


def test_valid_column_names_do_not_raise() -> None:
    validate_column_names(EXPECTED_COLUMNS)


def test_wrong_column_names_raise() -> None:
    invalid_columns = list(EXPECTED_COLUMNS)
    invalid_columns[-1] = "wrong_sensor"

    with pytest.raises(ValueError, match="Column names"):
        validate_column_names(invalid_columns)


def test_valid_trajectory_data_does_not_raise() -> None:
    rows = [
        [unit_id] + [1] * (len(EXPECTED_COLUMNS) - 1)
        for unit_id in range(1, 101)
    ]
    frame = pd.DataFrame(
        rows,
        columns=EXPECTED_COLUMNS,
    )

    validate_trajectory_data(frame)


def test_trajectory_data_rejects_non_numeric_value() -> None:
    rows = [
        [unit_id] + [1] * (len(EXPECTED_COLUMNS) - 1)
        for unit_id in range(1, 101)
    ]
    frame = pd.DataFrame(
        rows,
        columns=EXPECTED_COLUMNS,
    )
    frame["sensor_1"] = frame["sensor_1"].astype(object)
    frame.loc[0, "sensor_1"] = "bad_sensor_value"

    with pytest.raises(
        ValueError,
        match="Trajectory data contains non-numeric values",
    ):
        validate_trajectory_data(frame)


def test_trajectory_data_rejects_wrong_engine_count() -> None:
    rows = [
        [unit_id] + [1] * (len(EXPECTED_COLUMNS) - 1)
        for unit_id in range(1, 4)
    ]
    frame = pd.DataFrame(rows, columns=EXPECTED_COLUMNS)

    with pytest.raises(ValueError, match="100 unique engines"):
        validate_trajectory_data(frame)


def test_trajectory_data_rejects_missing_engine_id() -> None:
    frame = _valid_trajectory_frame()
    frame.loc[frame["unit_id"] == 100, "unit_id"] = 101

    with pytest.raises(ValueError, match=r"Missing: \[100\]"):
        validate_trajectory_data(frame)


def test_trajectory_data_rejects_unexpected_engine_id() -> None:
    frame = _valid_trajectory_frame()
    frame.loc[frame["unit_id"] == 100, "unit_id"] = 101

    with pytest.raises(ValueError, match=r"unexpected: \[101\]"):
        validate_trajectory_data(frame)


def _valid_trajectory_frame() -> pd.DataFrame:
    rows = []
    for unit_id in range(1, 101):
        for cycle in range(1, 3):
            rows.append(
                [unit_id, cycle] + [1] * (len(EXPECTED_COLUMNS) - 2)
            )
    return pd.DataFrame(rows, columns=EXPECTED_COLUMNS)


def test_trajectory_data_requires_cycle_one_start() -> None:
    frame = _valid_trajectory_frame()
    frame.loc[0, "cycle"] = 2

    with pytest.raises(ValueError, match="must start at 1"):
        validate_trajectory_data(frame)


def test_trajectory_data_requires_strictly_increasing_cycles() -> None:
    frame = _valid_trajectory_frame()
    frame.loc[1, "cycle"] = 1

    with pytest.raises(ValueError, match="increase strictly"):
        validate_trajectory_data(frame)
