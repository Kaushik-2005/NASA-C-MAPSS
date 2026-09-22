from pathlib import Path

from src.training.train import _git_revision, _sha256


def test_sha256_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("EngineGuard", encoding="utf-8")

    assert _sha256(path) == _sha256(path)
    assert len(_sha256(path)) == 64


def test_git_revision_returns_a_traceable_value() -> None:
    revision = _git_revision()

    assert revision
