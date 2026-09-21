from pathlib import Path

from src.eda.profile import build_profile


def test_profile_uses_fixed_engine_partitions() -> None:
    profile = build_profile(Path("data/raw/train_FD001.txt"))

    assert len(profile["development"]) == 16656
    assert profile["development"]["unit_id"].nunique() == 80
    assert profile["validation"]["unit_id"].nunique() == 20


def test_profile_records_development_only_exclusions() -> None:
    profile = build_profile(Path("data/raw/train_FD001.txt"))

    assert set(profile["excluded_features"]) == {
        "sensor_1",
        "sensor_5",
        "sensor_10",
        "sensor_16",
        "sensor_18",
        "sensor_19",
    }
