"""Materialize Module 5 engine and sample manifests."""

import json
from pathlib import Path

from src.data.ingest import load_fd001_file
from src.data.labels import add_rul_labels
from src.data.leakage import assert_disjoint_partitions
from src.data.samples import (
    generate_final_test_samples,
    generate_training_samples,
    generate_validation_samples,
)
from src.data.splits import build_engine_manifests, write_engine_manifests


def materialize_manifests(
    train_path: Path = Path("data/raw/train_FD001.txt"),
    test_path: Path = Path("data/raw/test_FD001.txt"),
    output_dir: Path = Path("data/manifests"),
) -> None:
    """Build and write fixed engine and sample manifest artifacts."""
    train = add_rul_labels(load_fd001_file(train_path))
    test = load_fd001_file(test_path)
    manifests = build_engine_manifests(train["unit_id"].unique())
    assert_disjoint_partitions(manifests)
    write_engine_manifests(manifests, output_dir / "engine_manifests.json")

    training = generate_training_samples(train, manifests["development"])
    validation = generate_validation_samples(train, manifests["validation"])
    final_test = generate_final_test_samples(test)
    sample_manifest = {
        "minimum_history_cycles": 20,
        "training_samples": len(training),
        "validation_samples": len(validation.samples),
        "validation_requested_cutoffs": validation.requested_cutoffs,
        "validation_deduplicated_cutoffs": validation.deduplicated_cutoffs,
        "final_test_samples": len(final_test),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sample_manifest.json").write_text(
        json.dumps(sample_manifest, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    materialize_manifests()
    print("FD001 engine and sample manifests written")
