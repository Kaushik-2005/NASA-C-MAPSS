import pandas as pd

from src.data.samples import (
    generate_final_test_samples,
    generate_training_samples,
    generate_validation_samples,
)


def make_frame(engine_count: int = 3, cycles: int = 100) -> pd.DataFrame:
    rows = []
    for unit_id in range(1, engine_count + 1):
        for cycle in range(1, cycles + 1):
            rows.append({"unit_id": unit_id, "cycle": cycle, "target_rul": 0})
    return pd.DataFrame(rows)


def test_training_samples_start_at_minimum_history() -> None:
    samples = generate_training_samples(make_frame(), [1, 2])

    assert samples["cycle"].min() == 20
    assert samples["unit_id"].nunique() == 2
    assert len(samples) == 162


def test_validation_samples_use_fixed_lifetime_cutoffs() -> None:
    result = generate_validation_samples(make_frame(), [1, 2])

    assert result.requested_cutoffs == 8
    assert result.deduplicated_cutoffs == 0
    assert result.samples["cycle"].tolist() == [60, 70, 80, 90, 60, 70, 80, 90]


def test_validation_cutoffs_are_deduplicated() -> None:
    result = generate_validation_samples(
        make_frame(cycles=20),
        [1],
        fractions=(0.60, 0.70, 0.80, 0.90),
    )

    assert result.requested_cutoffs == 4
    assert result.deduplicated_cutoffs == 3
    assert result.samples["cycle"].tolist() == [20]


def test_final_test_samples_select_last_row_per_engine() -> None:
    samples = generate_final_test_samples(make_frame())

    assert samples["unit_id"].tolist() == [1, 2, 3]
    assert samples["cycle"].tolist() == [100, 100, 100]
