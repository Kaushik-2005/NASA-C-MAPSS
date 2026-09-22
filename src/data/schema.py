"""Schema definitions and validation for NASA C-MAPSS FD001 data."""

from collections.abc import Sequence

import pandas as pd

EXPECTED_COLUMNS: tuple[str, ...] = (
    "unit_id",
    "cycle",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
    "sensor_1",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_5",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_10",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_16",
    "sensor_17",
    "sensor_18",
    "sensor_19",
    "sensor_20",
    "sensor_21",
)
EXPECTED_ENGINE_IDS = frozenset(range(1, 101))


def validate_column_count(column_count: int) -> None:
    expected_count = len(EXPECTED_COLUMNS)
    if column_count != expected_count:
        raise ValueError(
            f"Expected {expected_count} columns, but got {column_count}. "
            f"Expected columns: {EXPECTED_COLUMNS}"
        )


def validate_column_names(columns: Sequence[str]) -> None:
    if tuple(columns) != EXPECTED_COLUMNS:
        raise ValueError(
            "Column names do not match the expected FD001 schema. "
            f"Expected {EXPECTED_COLUMNS}, received {tuple(columns)}."
        )


def validate_trajectory_data(frame: pd.DataFrame) -> None:
    """Validate semantic rules for an FD001 trajectory DataFrame."""
    if tuple(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            "Trajectory DataFrame columns do not match the expected FD001 schema. "
            f"Expected {EXPECTED_COLUMNS}, received {tuple(frame.columns)}."
        )

    numeric_frame = frame.apply(pd.to_numeric, errors="coerce")
    if numeric_frame.isna().any().any():
        raise ValueError("Trajectory data contains non-numeric values.")

    engine_count = frame["unit_id"].nunique()
    if engine_count != 100:
        raise ValueError(f"Expected 100 unique engines, received {engine_count}.")

    observed_engine_ids = set(frame["unit_id"].astype(int))
    missing_engine_ids = EXPECTED_ENGINE_IDS - observed_engine_ids
    unexpected_engine_ids = observed_engine_ids - EXPECTED_ENGINE_IDS
    if missing_engine_ids or unexpected_engine_ids:
        raise ValueError(
            "Engine IDs must be exactly 1 through 100. "
            f"Missing: {sorted(missing_engine_ids)}, "
            f"unexpected: {sorted(unexpected_engine_ids)}."
        )

    for unit_id, engine_frame in frame.groupby("unit_id", sort=False):
        cycles = engine_frame["cycle"]
        if cycles.iloc[0] != 1:
            raise ValueError(f"Engine {unit_id} cycles must start at 1.")
        if not cycles.diff().dropna().gt(0).all():
            raise ValueError(f"Engine {unit_id} cycles must increase strictly.")
