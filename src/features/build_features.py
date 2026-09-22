"""Leakage-safe temporal feature construction for EngineGuard."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_SCHEMA_VERSION = "v1"
MINIMUM_HISTORY_CYCLES = 20
ROLLING_WINDOWS = (5, 10, 20)
DELTA_LAG = 5
SLOPE_WINDOW = 20

OPERATIONAL_COLUMNS = (
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
)

SENSOR_COLUMNS = tuple(f"sensor_{index}" for index in range(1, 22))

EXCLUDED_COLUMNS = {
    "sensor_1",
    "sensor_5",
    "sensor_10",
    "sensor_16",
    "sensor_18",
    "sensor_19",
}

RETAINED_COLUMNS = OPERATIONAL_COLUMNS + tuple(
    column for column in SENSOR_COLUMNS if column not in EXCLUDED_COLUMNS
)


def _build_feature_names() -> tuple[str, ...]:
    names: list[str] = []
    for column in RETAINED_COLUMNS:
        names.append(f"{column}__current")
        for window in ROLLING_WINDOWS:
            names.extend(
                (
                    f"{column}__rolling_mean_{window}",
                    f"{column}__rolling_std_{window}",
                )
            )
        names.extend(
            (
                f"{column}__delta_{DELTA_LAG}",
                f"{column}__slope_{SLOPE_WINDOW}",
            )
        )
    names.extend(("cycle__current", "history_count"))
    return tuple(names)


FEATURE_COLUMNS = _build_feature_names()


def feature_metadata() -> dict[str, object]:
    """Return the serializable feature contract for schema version v1."""
    return {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_columns": list(FEATURE_COLUMNS),
        "retained_columns": list(RETAINED_COLUMNS),
        "excluded_columns": sorted(EXCLUDED_COLUMNS),
        "minimum_history_cycles": MINIMUM_HISTORY_CYCLES,
        "rolling_windows": list(ROLLING_WINDOWS),
        "delta_lag": DELTA_LAG,
        "slope_window": SLOPE_WINDOW,
    }


def write_feature_metadata(path: Path) -> None:
    """Write the ordered feature contract as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(feature_metadata(), indent=2) + "\n",
        encoding="utf-8",
    )


def build_features(history: pd.DataFrame) -> pd.DataFrame:
    """Build one ordered feature row from an engine history."""
    if len(history) < MINIMUM_HISTORY_CYCLES:
        raise ValueError("At least 20 history cycles are required.")

    if history["cycle"].duplicated().any():
        raise ValueError("History cycles must not contain duplicates.")

    if not history["cycle"].is_monotonic_increasing:
        raise ValueError("History cycles must increase strictly.")

    missing_columns = set(RETAINED_COLUMNS) - set(history.columns)

    if missing_columns:
        raise ValueError(f"History is missing required columns: {sorted(missing_columns)}")

    return pd.DataFrame([_build_feature_dict(history)], columns=FEATURE_COLUMNS)


def _build_feature_dict(history: pd.DataFrame) -> dict[str, float | int]:
    latest = history.iloc[-1]
    features: dict[str, float | int] = {}
    for column in RETAINED_COLUMNS:
        values = history[column]
        features[f"{column}__current"] = values.iloc[-1]

        for window in ROLLING_WINDOWS:
            recent_values = values.tail(window)
            features[f"{column}__rolling_mean_{window}"] = recent_values.mean()
            features[f"{column}__rolling_std_{window}"] = recent_values.std(ddof=0)

        features[f"{column}__delta_{DELTA_LAG}"] = values.iloc[-1] - values.iloc[-DELTA_LAG - 1]

        slope_values = values.tail(SLOPE_WINDOW).to_numpy(dtype=float)
        slope_cycles = history["cycle"].tail(SLOPE_WINDOW).to_numpy(dtype=float)
        centered_cycles = slope_cycles - slope_cycles.mean()
        centered_values = slope_values - slope_values.mean()
        features[f"{column}__slope_{SLOPE_WINDOW}"] = float(
            np.dot(centered_cycles, centered_values) / np.dot(centered_cycles, centered_cycles)
        )

    features["cycle__current"] = latest["cycle"]
    features["history_count"] = min(len(history), MINIMUM_HISTORY_CYCLES)

    return features


def build_feature_matrix(
    trajectories: pd.DataFrame,
    samples: pd.DataFrame,
) -> pd.DataFrame:
    """Build one feature row per sample using only history up to its cycle."""
    required_sample_columns = {"unit_id", "cycle"}
    missing_sample_columns = required_sample_columns - set(samples.columns)
    if missing_sample_columns:
        raise ValueError(f"Samples are missing required columns: {sorted(missing_sample_columns)}")

    required_history_columns = {"unit_id", "cycle", *RETAINED_COLUMNS}
    missing_history_columns = required_history_columns - set(trajectories.columns)
    if missing_history_columns:
        raise ValueError(
            f"Trajectories are missing required columns: {sorted(missing_history_columns)}"
        )

    grouped_histories = {
        unit_id: engine.sort_values("cycle")
        for unit_id, engine in trajectories.groupby("unit_id", sort=False)
    }
    rows: list[dict[str, float | int]] = [{} for _ in range(len(samples))]

    for sample_positions, sample_group in samples.groupby("unit_id", sort=False):
        if sample_positions not in grouped_histories:
            raise ValueError(f"No trajectory found for engine {sample_positions}.")
        engine = grouped_histories[sample_positions]
        cycle_values = engine["cycle"].to_numpy()
        sample_cycles = sample_group["cycle"].to_numpy()
        end_indices = cycle_values.searchsorted(sample_cycles, side="right")
        if any(
            end_index == 0 or cycle_values[end_index - 1] != cycle
            for end_index, cycle in zip(end_indices, sample_cycles)
        ):
            raise ValueError(f"No trajectory row found for engine {sample_positions!r}.")

        positions = sample_group.index.to_numpy()
        for column in RETAINED_COLUMNS:
            values = engine[column].to_numpy(dtype=float)
            for position, end_index in zip(positions, end_indices):
                latest_index = end_index - 1
                rows[position][f"{column}__current"] = values[latest_index]
                for window in ROLLING_WINDOWS:
                    window_values = values[end_index - window : end_index]
                    rows[position][f"{column}__rolling_mean_{window}"] = float(window_values.mean())
                    rows[position][f"{column}__rolling_std_{window}"] = float(window_values.std())
                rows[position][f"{column}__delta_{DELTA_LAG}"] = (
                    values[latest_index] - values[latest_index - DELTA_LAG]
                )
                slope_cycles = cycle_values[end_index - SLOPE_WINDOW : end_index].astype(float)
                slope_values = values[end_index - SLOPE_WINDOW : end_index]
                centered_cycles = slope_cycles - slope_cycles.mean()
                centered_values = slope_values - slope_values.mean()
                rows[position][f"{column}__slope_{SLOPE_WINDOW}"] = float(
                    np.dot(centered_cycles, centered_values)
                    / np.dot(centered_cycles, centered_cycles)
                )
        for position, end_index in zip(positions, end_indices):
            rows[position]["cycle__current"] = int(cycle_values[end_index - 1])
            rows[position]["history_count"] = min(end_index, MINIMUM_HISTORY_CYCLES)

    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)
