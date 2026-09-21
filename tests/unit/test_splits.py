from src.data.splits import build_engine_manifests


def fd001_manifests() -> dict[str, set[int]]:
    return build_engine_manifests(range(1, 101))


def test_fd001_manifest_sizes() -> None:
    manifests = fd001_manifests()

    assert len(manifests["validation"]) == 20
    assert len(manifests["development"]) == 80
    assert len(manifests["initial_training"]) == 60
    assert len(manifests["incremental_training"]) == 20


def test_development_and_validation_are_disjoint() -> None:
    manifests = fd001_manifests()

    assert manifests["development"].isdisjoint(manifests["validation"])


def test_retraining_partitions_equal_development() -> None:
    manifests = fd001_manifests()

    combined = (
        manifests["initial_training"]
        | manifests["incremental_training"]
    )

    assert combined == manifests["development"]


def test_validation_ids_are_multiples_of_five() -> None:
    manifests = fd001_manifests()

    assert all(unit_id % 5 == 0 for unit_id in manifests["validation"])


def test_initial_and_incremental_training_are_disjoint() -> None:
    manifests = fd001_manifests()

    assert manifests["initial_training"].isdisjoint(
        manifests["incremental_training"]
    )
