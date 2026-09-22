# Project Tracker

## Current Module

- Module: 9 - Final Test Evaluation and Explainability
- Status: IN PROGRESS
- Started: 2026-09-21
- Last updated: 2026-09-22
- Current task: —
- Next action: Read Module 10 reproducibility and registry contract
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
| 10-17 | NOT STARTED | - | - |

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
| `.venv\\Scripts\\python.exe -m src.training.tree_benchmark` | 30 grouped trials completed; report written; MLflow experiment `engineguard-tree-models` logged | 2026-09-22 |
| `.venv\\Scripts\\python.exe -m pytest tests\\unit -q --tb=short` | 80 passed | 2026-09-22 |
| `.venv\\Scripts\\python.exe -m ruff check src\\training\\trees.py src\\training\\tree_benchmark.py tests\\unit\\test_trees.py` | All checks passed | 2026-09-22 |
| `.venv\\Scripts\\python.exe -m ruff check src\\evaluation\\metrics.py src\\training\\baselines.py tests\\unit\\test_baselines.py` | All checks passed | 2026-09-22 |
| `.venv\\Scripts\\python.exe -m src.training.benchmark` | Baseline report written; MLflow experiment `engineguard-baselines` logged | 2026-09-22 |
| `.venv\\Scripts\\python.exe -m ruff check src\\training\\benchmark.py` | All checks passed | 2026-09-22 |
| `.venv\\Scripts\\python.exe -m pytest tests\\unit -q --tb=short` | 75 passed | 2026-09-22 |

## Blockers and Risks

- Official test data must not influence feature selection or variance analysis; use development engines only.
