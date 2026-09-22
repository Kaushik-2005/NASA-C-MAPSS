import pandas as pd

from src.data.labels import add_rul_labels


def make_training_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unit_id": [1, 1, 1, 2, 2],
            "cycle": [1, 2, 31, 1, 150],
            "sensor_1": [10.0, 11.0, 12.0, 20.0, 21.0],
        }
    )


def test_labels_use_engine_specific_maximum_cycle() -> None:
    frame = make_training_fixture()

    labeled = add_rul_labels(frame)

    engine_one = labeled[labeled["unit_id"] == 1]
    engine_two = labeled[labeled["unit_id"] == 2]

    assert engine_one["raw_rul"].tolist() == [30, 29, 0]
    assert engine_two["raw_rul"].tolist() == [149, 0]


def test_target_rul_is_capped_at_125() -> None:
    frame = make_training_fixture()

    labeled = add_rul_labels(frame)

    engine_two_first_row = labeled[(labeled["unit_id"] == 2) & (labeled["cycle"] == 1)]

    assert engine_two_first_row["raw_rul"].iloc[0] == 149
    assert engine_two_first_row["target_rul"].iloc[0] == 125


def test_failure_label_includes_exact_30_cycle_boundary() -> None:
    frame = make_training_fixture()

    labeled = add_rul_labels(frame)

    engine_one_first_row = labeled[(labeled["unit_id"] == 1) & (labeled["cycle"] == 1)]

    assert engine_one_first_row["raw_rul"].iloc[0] == 30
    assert engine_one_first_row["failure_within_30"].iloc[0] == 1


def test_failure_label_is_zero_above_30_cycles() -> None:
    frame = make_training_fixture()

    labeled = add_rul_labels(frame)

    engine_two_first_row = labeled[(labeled["unit_id"] == 2) & (labeled["cycle"] == 1)]

    assert engine_two_first_row["raw_rul"].iloc[0] == 149
    assert engine_two_first_row["failure_within_30"].iloc[0] == 0


def test_labeling_does_not_mutate_original_frame() -> None:
    frame = make_training_fixture()
    original_columns = list(frame.columns)

    add_rul_labels(frame)

    assert list(frame.columns) == original_columns
    assert "raw_rul" not in frame.columns
    assert "target_rul" not in frame.columns
    assert "failure_within_30" not in frame.columns
