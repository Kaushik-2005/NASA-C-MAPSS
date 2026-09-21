from pathlib import Path

import pytest

from src.data.ingest import load_fd001_file, load_rul_file
from src.data.schema import EXPECTED_COLUMNS


def test_load_fd001_file_uses_expected_column_names(tmp_path: Path) -> None:
    path = tmp_path / "train_FD001.txt"
    rows = [
        " ".join(str(value) for value in [unit_id] + list(range(1, 26)))
        for unit_id in range(1, 101)
    ]
    path.write_text("\n".join(rows) + "\n")

    frame = load_fd001_file(path)

    assert frame.shape == (100, 26)
    assert list(frame.columns) == list(EXPECTED_COLUMNS)


def test_load_fd001_file_rejects_wrong_field_count(tmp_path: Path) -> None:
    path = tmp_path / "invalid.txt"
    path.write_text(" ".join(str(value) for value in range(25)) + "\n")

    with pytest.raises(ValueError, match="26"):
        load_fd001_file(path)


def test_load_rul_file_requires_100_values(tmp_path: Path) -> None:
    path = tmp_path / "RUL_FD001.txt"
    path.write_text("\n".join(str(value) for value in range(100)) + "\n")

    values = load_rul_file(path)

    assert len(values) == 100
    assert values.dtype.kind in {"i", "u"}


def test_load_rul_file_rejects_wrong_value_count(tmp_path: Path) -> None:
    path = tmp_path / "invalid_rul.txt"
    path.write_text("\n".join(str(value) for value in range(99)) + "\n")

    with pytest.raises(ValueError, match="100"):
        load_rul_file(path)
