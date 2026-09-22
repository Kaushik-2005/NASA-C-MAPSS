from pathlib import Path

from src.data.validate_raw import build_quality_report, validate_raw_files
from src.eda.profile import build_profile, write_profile


def test_raw_quality_report_contains_fd001_contract() -> None:
    validate_raw_files()
    report = build_quality_report()

    assert report["validation_status"] == "passed"
    assert report["files"]["train_FD001.txt"]["unique_engines"] == 100
    assert report["files"]["test_FD001.txt"]["unique_engines"] == 100
    assert report["files"]["RUL_FD001.txt"]["values"] == 100


def test_profile_build_and_write(tmp_path: Path) -> None:
    profile = build_profile()
    report_path = tmp_path / "profile.html"
    write_profile(report_path)

    assert len(profile["excluded_features"]) == 6
    assert "NASA C-MAPSS FD001 profile" in report_path.read_text(encoding="utf-8")
