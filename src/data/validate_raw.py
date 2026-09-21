"""Command-line validation and reporting for immutable FD001 raw files."""

import argparse
import json
from pathlib import Path

from src.data.checksums import verify_sha256
from src.data.ingest import load_fd001_file, load_rul_file


RAW_DIR = Path("data/raw")
EXPECTED_SHA256 = {
    "train_FD001.txt": "963B5E22825B34D8B21C69E1AEB4AF3E647050EB672EE8834BA4B5D91D2DE0F8",
    "test_FD001.txt": "3CDA7109CE17BAFB5443F2AC926CFCF88154B941B8C4CF95EB55D1DDD6F52851",
    "RUL_FD001.txt": "A19C8EC94931949D0485BDC35118206E9C81C4547B422EFB9CF86F4CEDDBCECA",
}


def validate_raw_files(raw_dir: Path = RAW_DIR) -> None:
    """Validate checksums and semantic structure of all required FD001 files."""
    for name, checksum in EXPECTED_SHA256.items():
        verify_sha256(raw_dir / name, checksum)

    train = load_fd001_file(raw_dir / "train_FD001.txt")
    test = load_fd001_file(raw_dir / "test_FD001.txt")
    rul = load_rul_file(raw_dir / "RUL_FD001.txt")
    if train.empty or test.empty or len(rul) != 100:
        raise ValueError("FD001 raw files contain no valid data.")


def build_quality_report(raw_dir: Path = RAW_DIR) -> dict[str, object]:
    """Build a machine-readable quality report after validation succeeds."""
    validate_raw_files(raw_dir)
    report: dict[str, object] = {
        "dataset": "FD001",
        "validation_status": "passed",
        "files": {},
    }

    train = load_fd001_file(raw_dir / "train_FD001.txt")
    test = load_fd001_file(raw_dir / "test_FD001.txt")
    rul = load_rul_file(raw_dir / "RUL_FD001.txt")

    for name, frame in (("train_FD001.txt", train), ("test_FD001.txt", test)):
        report["files"][name] = {
            "sha256": EXPECTED_SHA256[name],
            "rows": int(len(frame)),
            "columns": int(frame.shape[1]),
            "unique_engines": int(frame["unit_id"].nunique()),
            "missing_values": int(frame.isna().sum().sum()),
            "cycle_min": int(frame["cycle"].min()),
            "cycle_max": int(frame["cycle"].max()),
        }

    report["files"]["RUL_FD001.txt"] = {
        "sha256": EXPECTED_SHA256["RUL_FD001.txt"],
        "values": int(len(rul)),
        "missing_values": int(rul.isna().sum()),
        "min_rul": int(rul.min()),
        "max_rul": int(rul.max()),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional JSON path for the quality report.",
    )
    args = parser.parse_args()
    report = build_quality_report()
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("FD001 raw-file validation passed")


if __name__ == "__main__":
    main()
