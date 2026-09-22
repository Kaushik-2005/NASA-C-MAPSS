"""Small repository checks for accidental secrets and unsafe serving files."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".toml", ".yaml", ".yml", ".json", ".md", ".txt"}
TEXT_FILENAMES = {"Dockerfile", ".dockerignore"}
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"(?:AKIA|ASIA)[0-9A-Z]{16}"),
    re.compile(r"(?i)(?:api[_-]?key|secret|password)\s*[=:]\s*[\"'][^\"']{12,}[\"']"),
)
ALLOWED_SERVING_ARTIFACTS = {
    "models/xgboost_candidate_v1.joblib",
    "models/feature_preprocessor_v1.joblib",
    "models/champion_initial_v1.joblib",
    "models/feature_preprocessor_initial_v1.joblib",
    "models/challenger_incremental_v1.joblib",
    "models/feature_preprocessor_challenger_v1.joblib",
}


def scan_text_files() -> list[str]:
    """Return relative paths containing a high-confidence secret pattern."""
    findings: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or (
            path.suffix.lower() not in TEXT_SUFFIXES and path.name not in TEXT_FILENAMES
        ):
            continue
        if any(part in {".git", ".venv", "__pycache__", ".mypy_cache", ".pytest_cache"} for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            findings.append(path.relative_to(ROOT).as_posix())
    return findings


def check_serving_allowlist() -> list[str]:
    """Return unexpected joblib artifacts under the serving model directory."""
    model_dir = ROOT / "models"
    if not model_dir.exists():
        return []
    return [
        path.relative_to(ROOT).as_posix()
        for path in model_dir.glob("*.joblib")
        if path.relative_to(ROOT).as_posix() not in ALLOWED_SERVING_ARTIFACTS
    ]


def main() -> int:
    secrets = scan_text_files()
    unexpected_models = check_serving_allowlist()
    if secrets or unexpected_models:
        if secrets:
            print(f"Potential secrets found: {secrets}")
        if unexpected_models:
            print(f"Unexpected serving artifacts found: {unexpected_models}")
        return 1
    print("Security audit passed: no high-confidence secrets or unexpected serving artifacts found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
