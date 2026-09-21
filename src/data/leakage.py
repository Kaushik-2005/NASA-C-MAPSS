"""Assertions protecting engine, temporal, and target leakage boundaries."""

from collections.abc import Mapping, Sequence, Set

import pandas as pd


FORBIDDEN_FEATURE_COLUMNS = frozenset(
    {
        "raw_rul",
        "target_rul",
        "failure_within_30",
        "maximum_cycle",
        "max_training_cycle",
    }
)


def assert_disjoint_partitions(manifests: Mapping[str, Set[int]]) -> None:
    """Ensure no engine appears in more than one partition."""
    pairs = [("development", "validation"), ("initial_training", "incremental_training")]
    available_pairs = [pair for pair in pairs if all(name in manifests for name in pair)]
    if not available_pairs:
        names = list(manifests)
        available_pairs = [
            (left, right)
            for index, left in enumerate(names)
            for right in names[index + 1 :]
        ]

    for left_name, right_name in available_pairs:
        overlap = set(manifests[left_name]) & set(manifests[right_name])
        if overlap:
            raise AssertionError(
                f"Partitions {left_name} and {right_name} overlap: {sorted(overlap)}"
            )


def assert_feature_columns_are_safe(columns: Sequence[str]) -> None:
    """Ensure labels and lifecycle-derived values are absent from features."""
    forbidden = set(columns) & FORBIDDEN_FEATURE_COLUMNS
    if forbidden:
        raise AssertionError(f"Forbidden leakage columns found: {sorted(forbidden)}")


def assert_samples_belong_to_engines(
    samples: pd.DataFrame,
    engine_ids: Set[int],
) -> None:
    """Ensure generated samples contain only engines from their manifest."""
    observed = set(samples["unit_id"])
    unexpected = observed - set(engine_ids)
    if unexpected:
        raise AssertionError(f"Samples contain unexpected engines: {sorted(unexpected)}")
