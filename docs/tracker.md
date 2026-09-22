# Project Tracker

## Current Module

- Module: 12 - Tests, Docker, and CI
- Status: COMPLETE
- Started: 2026-09-21
- Last updated: 2026-09-22
- Current task: —
- Next action: Start Module 13 - Fixed AWS Deployment
- Blockers: None

## Module Status

| Module | Status | Evidence | Notes |
| --- | --- | --- | --- |
| 1 | COMPLETE | `pyproject.toml`, `Makefile`, `docs/`, `tests/test_smoke.py` | Smoke gate passed |
| 2 | IMPLEMENTED - VERIFICATION PENDING | `src/foundations/`, `docs/ml-fundamentals.md`, `notebooks/01_ml_foundations.ipynb` | NumPy tests pass; FD001-backed notebook comparison remains |
| 3 | IMPLEMENTED - VERIFICATION PENDING | `data/raw/`, `src/data/`, `data/checksums.sha256`, `dvc.yaml`, `reports/fd001-data-quality.json` | 26 tests and raw validation pass; DVC lock refresh pending |
| 4 | COMPLETE | `notebooks/02_fd001_eda.ipynb`, `docs/data-dictionary.md`, `reports/fd001-profile.html`, `configs/excluded_features.json` | 28 tests pass; profile generated |
| 5 | COMPLETE | `src/data/labels.py`, `src/data/splits.py`, `src/data/samples.py`, `src/data/leakage.py`, `docs/leakage-analysis.md`, `data/manifests/` | 48 tests pass; fixed samples and leakage checks verified |
| 6 | COMPLETE | `src/features/`, `configs/feature_schema_v1.json`, `models/feature_preprocessor_v1.joblib`, `reports/feature-pipeline-v1.json`, `docs/design.md` | 64 tests pass; development artifact fitted |
| 7 | COMPLETE | `src/training/benchmark.py`, `reports/baseline-results.md`, `reports/baseline-results.json`, MLflow experiment `engineguard-baselines` | Validation benchmark, timing, model size, and threshold comparison verified |
| 8 | COMPLETE | `src/training/trees.py`, `src/training/tree_benchmark.py`, `reports/tree-model-comparison.md`, `configs/xgboost-candidate-v1.json`, MLflow experiment `engineguard-tree-models` | 30 grouped trials, validation comparison, timing, size, and frozen candidate verified |
| 9 | COMPLETE | `src/evaluation/final_evaluation.py`, `reports/final-evaluation.md`, `reports/final-evaluation.json`, `reports/final-test-predictions.csv`, `reports/final-explainability/`, `docs/model-card.md` | One-time 100-engine holdout evaluation and SHAP artifacts verified |
| 10 | COMPLETE | `src/training/train.py`, `dvc.yaml`, `reports/training-lineage.json`, MLflow model `EngineGuardRUL` versions 1-2 | DVC repro passed; lineage, registration, and aliases verified |
| 11 | COMPLETE | `src/api/`, `tests/unit/test_api_contract.py` | All endpoints, validation, readiness, parity, and OpenAPI smoke checks verified |
| 12 | COMPLETE | `Dockerfile`, `.github/workflows/ci.yml`, `.dockerignore`, `coverage.xml`, targeted coverage tests | 101 tests passed; 81.67% `src/` coverage; Docker smoke passed |
| 13-17 | NOT STARTED | - | - |

## Module 12 Checklist

- [x] Add non-root multi-stage Dockerfile
- [x] Add container health check
- [x] Add CI workflow for lint, typing, tests, coverage, training, and Docker build
- [x] Add dependency audit step
- [x] Add 100-engine batch-size test
- [x] Run full functional test suite
- [x] Run repository Ruff checks
- [x] Reach 80% `src/` coverage
- [x] Build and smoke-test the container
- [x] Run Module 12 completion gate

## Module 11 Checklist

- [x] Add FastAPI and Uvicorn dependencies
- [x] Define typed observation and request contracts
- [x] Define fixed prediction response contract
- [x] Load champion model once during startup
- [x] Implement all five required endpoints
- [x] Validate minimum history and strictly increasing cycles
- [x] Enforce bounded predictions and deterministic risk mapping
- [x] Add request IDs and structured prediction logging
- [x] Add API contract tests
- [x] Add offline/online prediction parity test
- [x] Run serving smoke and readiness gate

## Module 10 Checklist

- [x] Implement reproducible training entry point
- [x] Record data, DVC, Git, feature, config, metric, and model lineage
- [x] Add DVC training stage
- [x] Connect `make train`
- [x] Register MLflow model version
- [x] Assign `candidate` and `champion` aliases
- [x] Add reproducibility and lineage tests
- [x] Verify `dvc repro train` from an activated user shell
- [x] Run Module 10 completion gate

## Module 9 Checklist

- [x] Freeze the XGBoost candidate configuration
- [x] Retrain the candidate on all development engines
- [x] Produce exactly 100 official test predictions
- [x] Enforce prediction bounds 0-125
- [x] Report RMSE, MAE, R-squared, and NASA asymmetric score
- [x] Report accuracy windows and true-RUL error bands
- [x] Inspect largest underpredictions and overpredictions
- [x] Generate global and three local SHAP explanations
- [x] Write the model card and error-analysis notebook
- [x] Run Module 9 completion gate

## Module 8 Checklist

- [x] Add XGBoost dependency and verify installation
- [x] Implement Random Forest and XGBoost builders
- [x] Implement five-fold engine-grouped cross-validation
- [x] Freeze the exact 30-configuration search contract
- [x] Add grouped-split and search-contract tests
- [x] Train Random Forest on development samples
- [x] Run the 30-configuration XGBoost search
- [x] Compare Ridge, Random Forest, and XGBoost on fixed validation samples
- [x] Write the model-comparison report
- [x] Log Module 8 trials to MLflow
- [x] Run Module 8 completion gate

## Current Module Checklist

- [x] Implement RMSE, MAE, and R-squared
- [x] Implement median constant predictor
- [x] Implement Ridge Regression baseline
- [x] Implement Logistic Regression learning artifact
- [x] Add regression and classification tests
- [x] Measure time and serialized model size
- [x] Log baseline runs to MLflow
- [x] Write baseline report
- [x] Run Module 7 completion gate

- [x] Define retained feature columns and schema version `v1`
- [x] Implement current, rolling, delta, and slope features
- [x] Add cycle/history features
- [x] Enforce minimum history of 20 cycles
- [x] Add temporal leakage and boundary tests
- [x] Add offline/online parity test
- [x] Run Module 6 completion gate

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
| `.venv\\Scripts\\python.exe -m pytest tests\\unit\\test_baselines.py -q` | 4 passed | 2026-09-21 |
| `.venv\\Scripts\\python.exe -m pytest tests\\unit -q --tb=short` | 68 passed | 2026-09-21 |
| `.venv\\Scripts\\python.exe -m pytest tests\\unit\\test_baselines.py -q --tb=short` | 7 passed | 2026-09-21 |
| `.venv\\Scripts\\python.exe -m pytest tests\\unit -q --tb=short` | 71 passed | 2026-09-21 |
| `.venv\\Scripts\\python.exe -m pytest tests\\unit -q --tb=short` | 75 passed | 2026-09-22 |
| `.venv\\Scripts\\python.exe -m src.evaluation.final_evaluation` | 100 bounded predictions; final artifacts and MLflow run written | 2026-09-22 |
| `.venv\\Scripts\\python.exe -m pytest tests\\unit\\test_final_evaluation.py -q --tb=short` | 3 passed | 2026-09-22 |
| `.venv\\Scripts\\python.exe -m pytest tests\\unit -q --tb=short` | 83 passed | 2026-09-22 |
| `.venv\\Scripts\\python.exe -m ruff check src\\evaluation\\final_evaluation.py tests\\unit\\test_final_evaluation.py` | All checks passed | 2026-09-22 |
| `.venv\\Scripts\\python.exe -m src.training.train` | Registered `EngineGuardRUL` version 1 | 2026-09-22 |
| MLflow alias lookup | `candidate=1`, `champion=1` | 2026-09-22 |
| `.venv\\Scripts\\python.exe -m pytest tests\\unit -q --tb=short` | 85 passed | 2026-09-22 |
| `.venv\\Scripts\\python.exe -m ruff check src\\training\\train.py tests\\unit\\test_train.py` | All checks passed | 2026-09-22 |
| `.venv\\Scripts\\python.exe -m dvc dag` | `validate_raw` and `train` stages recognized with workspace-local DVC config | 2026-09-22 |
| `dvc repro train` | Stage completed; `EngineGuardRUL` version 2 registered; `dvc.lock` updated | 2026-09-22 |
| `.venv\\Scripts\\python.exe -m pytest tests\\unit\\test_api_contract.py -q` | 4 passed | 2026-09-22 |
| `.venv\\Scripts\\python.exe -m pytest tests\\unit -q --tb=short` | 89 passed | 2026-09-22 |
| `.venv\\Scripts\\python.exe -m ruff check src\\api tests\\unit\\test_api_contract.py` | All checks passed | 2026-09-22 |
| `.venv\\Scripts\\python.exe -m pytest tests\\unit\\test_api_contract.py -q` | 7 passed | 2026-09-22 |
| `.venv\\Scripts\\python.exe -m pytest --cov=src --cov-report=term-missing --cov-report=xml --cov-fail-under=0` | 93 passed; 72% coverage | 2026-09-22 |
| `.venv\\Scripts\\python.exe -m ruff format --check src tests` | Pending after formatting pass | 2026-09-22 |
| `.venv\\Scripts\\python.exe -m ruff check src tests` | Pending final verification | 2026-09-22 |
| `docker build --tag engineguard-local:module12 .` | Blocked by Docker daemon permission on `.docker/buildx/instances` | 2026-09-22 |
| `.venv\\Scripts\\python.exe -m pytest --cov=src --cov-report=term-missing --cov-report=xml --cov-fail-under=80` | 101 passed; 81.67% coverage | 2026-09-22 |
| `.venv\\Scripts\\python.exe -m ruff check src tests` | All checks passed | 2026-09-22 |
| `.venv\\Scripts\\python.exe -m ruff format --check src tests` | 52 files already formatted | 2026-09-22 |
| `docker build --tag engineguard-local:module12 .` | Image built successfully | 2026-09-22 |
| Container smoke: `/health`, `/ready`, `/model-info`, `/v1/predict` | Health/readiness/metadata/prediction checks passed | 2026-09-22 |
| `.venv\\Scripts\\python.exe -m pytest tests\\unit\\test_api_contract.py -q` | 6 passed | 2026-09-22 |
| `.venv\\Scripts\\python.exe -m pytest tests\\unit -q --tb=short` | 91 passed | 2026-09-22 |
| `.venv\\Scripts\\python.exe -m ruff check src\\api tests\\unit\\test_api_contract.py` | All checks passed | 2026-09-22 |
| `.venv\\Scripts\\python.exe -m src.training.tree_benchmark` | 30 grouped trials completed; report written; MLflow experiment `engineguard-tree-models` logged | 2026-09-22 |
| `.venv\\Scripts\\python.exe -m pytest tests\\unit -q --tb=short` | 80 passed | 2026-09-22 |
| `.venv\\Scripts\\python.exe -m ruff check src\\training\\trees.py src\\training\\tree_benchmark.py tests\\unit\\test_trees.py` | All checks passed | 2026-09-22 |
| `.venv\\Scripts\\python.exe -m ruff check src\\evaluation\\metrics.py src\\training\\baselines.py tests\\unit\\test_baselines.py` | All checks passed | 2026-09-22 |
| `.venv\\Scripts\\python.exe -m src.training.benchmark` | Baseline report written; MLflow experiment `engineguard-baselines` logged | 2026-09-22 |
| `.venv\\Scripts\\python.exe -m ruff check src\\training\\benchmark.py` | All checks passed | 2026-09-22 |
| `.venv\\Scripts\\python.exe -m pytest tests\\unit -q --tb=short` | 75 passed | 2026-09-22 |

## Blockers and Risks

- Official test data must not influence feature selection or variance analysis; use development engines only.
