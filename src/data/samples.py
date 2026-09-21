"""Fixed training, validation, and final-test sample policies."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import pandas as pd


MIN_HISTORY_CYCLES = 20
VALIDATION_FRACTIONS = (0.60, 0.70, 0.80, 0.90)


@dataclass(frozen=True)
class ValidationSampleResult:
    """Validation samples plus cutoff-deduplication evidence."""

    samples: pd.DataFrame
    requested_cutoffs: int
    deduplicated_cutoffs: int


def generate_training_samples(
    frame: pd.DataFrame,
    engine_ids: Iterable[int],
    minimum_history: int = MIN_HISTORY_CYCLES,
) -> pd.DataFrame:
    """Return every eligible labeled training row from cycle 20 onward."""
    _validate_minimum_history(minimum_history)
    selected = frame[
        frame["unit_id"].isin(set(engine_ids))
        & (frame["cycle"] >= minimum_history)
    ]
    return selected.sort_values(["unit_id", "cycle"]).reset_index(drop=True)


def generate_validation_samples(
    frame: pd.DataFrame,
    engine_ids: Iterable[int],
    fractions: Sequence[float] = VALIDATION_FRACTIONS,
    minimum_history: int = MIN_HISTORY_CYCLES,
) -> ValidationSampleResult:
    """Return one sample at each fixed lifetime fraction per validation engine."""
    _validate_minimum_history(minimum_history)
    if not fractions:
        raise ValueError("fractions must contain at least one cutoff")
    if any(fraction <= 0 or fraction > 1 for fraction in fractions):
        raise ValueError("fractions must be greater than 0 and at most 1")

    rows: list[pd.Series] = []
    requested_cutoffs = 0
    deduplicated_cutoffs = 0
    selected_ids = set(engine_ids)

    for unit_id, engine in frame[frame["unit_id"].isin(selected_ids)].groupby(
        "unit_id", sort=True
    ):
        engine = engine.sort_values("cycle")
        maximum_cycle = int(engine["cycle"].max())
        seen_cycles: set[int] = set()
        for fraction in fractions:
            requested_cutoffs += 1
            cutoff = max(minimum_history, int(maximum_cycle * fraction))
            if cutoff in seen_cycles:
                deduplicated_cutoffs += 1
                continue
            seen_cycles.add(cutoff)
            match = engine[engine["cycle"] == cutoff]
            if match.empty:
                raise ValueError(
                    f"Engine {unit_id} has no row at validation cutoff {cutoff}."
                )
            rows.append(match.iloc[0])

    samples = (
        pd.DataFrame(rows, columns=frame.columns)
        .sort_values(["unit_id", "cycle"])
        .reset_index(drop=True)
    )
    return ValidationSampleResult(
        samples=samples,
        requested_cutoffs=requested_cutoffs,
        deduplicated_cutoffs=deduplicated_cutoffs,
    )


def generate_final_test_samples(
    frame: pd.DataFrame,
    minimum_history: int = MIN_HISTORY_CYCLES,
) -> pd.DataFrame:
    """Return the final available row for each official test engine."""
    _validate_minimum_history(minimum_history)
    eligible = frame[frame["cycle"] >= minimum_history]
    samples = (
        eligible.sort_values(["unit_id", "cycle"])
        .groupby("unit_id", sort=True, as_index=False)
        .tail(1)
        .sort_values("unit_id")
        .reset_index(drop=True)
    )
    if samples["unit_id"].nunique() != frame["unit_id"].nunique():
        raise ValueError("Every test engine must have at least 20 history cycles.")
    return samples


def _validate_minimum_history(minimum_history: int) -> None:
    if minimum_history < 1:
        raise ValueError("minimum_history must be positive")
