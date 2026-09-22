from pathlib import Path

from scripts.security_audit import check_serving_allowlist, scan_text_files


def test_repository_security_audit_has_no_high_confidence_secrets() -> None:
    assert scan_text_files() == []


def test_serving_artifact_allowlist_is_clean() -> None:
    assert check_serving_allowlist() == []


def test_audit_script_is_in_repository() -> None:
    assert Path("scripts/security_audit.py").exists()
