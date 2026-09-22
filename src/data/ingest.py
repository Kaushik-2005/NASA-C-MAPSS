"""Raw NASA C-MAPSS FD001 ingestion."""

from pathlib import Path

import pandas as pd

from src.data.schema import (
    EXPECTED_COLUMNS,
    validate_column_count,
    validate_trajectory_data,
)


def load_fd001_file(path: Path) -> pd.DataFrame:
    """Load a whitespace-separated FD001 trajectory file."""
    if not path.exists():
        raise FileNotFoundError(f"FD001 file not found: {path}")

    frame = pd.read_csv(path, sep=r"\s+", header=None)
    validate_column_count(frame.shape[1])
    frame.columns = EXPECTED_COLUMNS
    validate_trajectory_data(frame)
    return frame


def load_rul_file(path: Path) -> pd.Series:
    """Load and validate the 100 official FD001 test RUL values."""
    if not path.exists():
        raise FileNotFoundError(f"RUL file not found: {path}")

    frame = pd.read_csv(path, sep=r"\s+", header=None)
    if frame.shape[1] != 1:
        raise ValueError(f"Expected 1 RUL value per row, received {frame.shape[1]} columns.")
    if len(frame) != 100:
        raise ValueError(f"Expected 100 RUL values, received {len(frame)}.")

    values = pd.to_numeric(frame.iloc[:, 0], errors="coerce")
    if values.isna().any():
        raise ValueError("RUL file contains non-numeric values.")
    return values.astype(int)
