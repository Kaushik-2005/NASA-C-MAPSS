import hashlib
from pathlib import Path

import pytest

from src.data.checksums import sha256_file, verify_sha256


def test_sha256_file_matches_known_digest(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_bytes(b"engineguard")

    expected = hashlib.sha256(b"engineguard").hexdigest()

    assert sha256_file(path) == expected


def test_verify_sha256_accepts_uppercase_digest(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_bytes(b"engineguard")
    expected = hashlib.sha256(b"engineguard").hexdigest().upper()

    verify_sha256(path, expected)


def test_verify_sha256_rejects_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_bytes(b"engineguard")

    with pytest.raises(ValueError, match="Checksum mismatch"):
        verify_sha256(path, "0" * 64)
