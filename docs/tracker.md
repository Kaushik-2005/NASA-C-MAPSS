# Project Tracker

## Current Module

- Module: 6 - Fixed Temporal Feature Pipeline
- Status: NOT STARTED
- Started: 2026-09-21
- Last updated: 2026-09-21
- Current task: Prepare the shared temporal feature pipeline
- Next action: Read Module 6 contract and define ordered feature schema
- Blockers: Git is not initialized in the workspace; DVC lock refresh should be confirmed in the user's activated shell

## Module Status

| Module | Status | Evidence | Notes |
| --- | --- | --- | --- |
| 1 | COMPLETE | `pyproject.toml`, `Makefile`, `docs/`, `tests/test_smoke.py` | Smoke gate passed |
| 2 | IMPLEMENTED - VERIFICATION PENDING | `src/foundations/`, `docs/ml-fundamentals.md`, `notebooks/01_ml_foundations.ipynb` | NumPy tests pass; FD001-backed notebook comparison remains |
| 3 | IMPLEMENTED - VERIFICATION PENDING | `data/raw/`, `src/data/`, `data/checksums.sha256`, `dvc.yaml`, `reports/fd001-data-quality.json` | 26 tests and raw validation pass; DVC lock refresh pending |
| 4 | COMPLETE | `notebooks/02_fd001_eda.ipynb`, `docs/data-dictionary.md`, `reports/fd001-profile.html`, `configs/excluded_features.json` | 28 tests pass; profile generated |
| 5 | COMPLETE | `src/data/labels.py`, `src/data/splits.py`, `src/data/samples.py`, `src/data/leakage.py`, `docs/leakage-analysis.md`, `data/manifests/` | 48 tests pass; fixed samples and leakage checks verified |
| 6 | NOT STARTED | - | Shared temporal features next |
| 7-17 | NOT STARTED | - | - |

## Current Module Checklist

- [x] Implement `raw_rul`, `target_rul`, and `failure_within_30`
- [x] Add label tests
- [x] Materialize deterministic engine manifests
- [x] Generate fixed training and validation samples
- [x] Add leakage tests
- [x] Write leakage analysis
- [x] Run Module 5 completion gate

- [x] Create EDA notebook
- [x] Create development/validation engine split for EDA
- [x] Report row counts and engine counts
- [x] Plot lifecycle-length distribution and representative trajectories
- [x] Summarize operating settings
- [x] Compute development-only sensor variance
- [x] Identify constant and near-constant features
- [x] Write data dictionary and FD001 limitations
- [x] Run Module 4 completion gate

## Verification Evidence

| Command | Result | Date |
| --- | --- | --- |
| `.venv\\Scripts\\python.exe -m pytest tests/unit -q` | 26 passed | 2026-09-21 |
| `.venv\\Scripts\\python.exe -m src.data.validate_raw --report reports/fd001-data-quality.json` | FD001 raw-file validation passed | 2026-09-21 |
| `.venv\\Scripts\\python.exe -m dvc dag` | `validate_raw` stage recognized | 2026-09-21 |
| `.venv\\Scripts\\python.exe -m pytest tests/unit -q` | 28 passed | 2026-09-21 |
| `.venv\\Scripts\\python.exe -m src.eda.profile` | Wrote `reports/fd001-profile.html` | 2026-09-21 |

## Blockers and Risks

- Git metadata is absent in the current directory, so repository history cannot be inspected or updated.
- Official test data must not influence feature selection or variance analysis; use development engines only.
