"""Checksum utilities for immutable raw FD001 files."""

import hashlib
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the lowercase SHA-256 digest of a file."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: Path, expected: str) -> None:
    """Raise ValueError when a file does not match an expected SHA-256 digest."""
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise ValueError(
            f"Checksum mismatch for {path}: expected {expected}, received {actual}."
        )
