import json
from pathlib import Path

from src.data.validate_raw import build_quality_report


def test_quality_report_contains_fd001_file_measurements() -> None:
    report = build_quality_report(Path("data/raw"))

    assert report["dataset"] == "FD001"
    assert report["validation_status"] == "passed"
    files = report["files"]
    assert files["train_FD001.txt"]["rows"] == 20631
    assert files["train_FD001.txt"]["columns"] == 26
    assert files["train_FD001.txt"]["unique_engines"] == 100
    assert files["test_FD001.txt"]["rows"] == 13096
    assert files["RUL_FD001.txt"]["values"] == 100


def test_quality_report_is_json_serializable() -> None:
    report = build_quality_report(Path("data/raw"))

    encoded = json.dumps(report)

    assert '"validation_status": "passed"' in encoded
