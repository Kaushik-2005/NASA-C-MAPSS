"""Deterministic engine-level partitions for the FD001 workflow."""

from collections.abc import Iterable
import json
from pathlib import Path


def build_engine_manifests(
    engine_ids: Iterable[int],
) -> dict[str, set[int]]:
    """Build the fixed development, validation, and retraining partitions."""
    all_engine_ids = {int(engine_id) for engine_id in engine_ids}
    validation = {unit_id for unit_id in all_engine_ids if unit_id % 5 == 0}
    development = all_engine_ids - validation
    initial_training = {
        unit_id
        for unit_id in development
        if unit_id <= 75
    }
    incremental_training = {
        unit_id
        for unit_id in development
        if unit_id > 75
    }

    return {
        "development": development,
        "validation": validation,
        "initial_training": initial_training,
        "incremental_training": incremental_training,
    }


def write_engine_manifests(
    manifests: dict[str, set[int]],
    path: Path,
) -> None:
    """Write sorted engine manifests as a reproducible JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: sorted(engine_ids) for name, engine_ids in manifests.items()}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
