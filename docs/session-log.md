# Session Log

## 2026-09-21 — Module 1

- Goal: Start the roadmap and establish the repository foundation.
- Changes: Added package metadata, configs, task commands, typed fixed-contract config, problem/architecture docs, tracker, decision record, and smoke test.
- Tests and results: Direct fixed-contract assertions passed; `compileall` passed. Pytest could not start because the provisioned runtime has an incomplete/inaccessible `pygments` installation.
- Learning captured: ML and system boundaries are defined before model implementation; FD001 test data remains isolated.
- Decisions made: Accepted typed project constants as the initial contract boundary.
- Blockers: Git metadata and FD001 files are absent.
- Next action: Run the Module 1 pytest gate in a clean Python environment, then teach/build Module 2 foundations.

## 2026-09-21 — Module 2

- Goal: Build transparent NumPy foundations for regression and classification.
- Changes: Added batch-gradient Linear Regression and Logistic Regression, tests, foundations notes, and notebook scaffold.
- Tests and results: `.venv\\Scripts\\python.exe -m pytest -q` — 4 passed; compileall passed.
- Learning captured: MSE, log loss, L2 regularization, gradient descent, scaling, class imbalance, and engine-level leakage.
- Decisions made: Preserve NumPy implementations as learning artifacts; keep the derived classifier out of the deployed risk path.
- Blockers: FD001 raw files are absent, so the scikit-learn comparison, learning-rate plots, and FD001-specific evidence remain pending.
- Next action: Continue with Module 3 ingestion once data is available.

## 2026-09-21 — Module 3

- Goal: Obtain and verify the immutable FD001 raw files before ingestion coding.
- Changes: Added `data/raw/train_FD001.txt`, `test_FD001.txt`, and `RUL_FD001.txt`; recorded source and SHA-256 checksums in `docs/data-source.md`.
- Tests and results: Train/test have 100 engines and 26 non-empty fields per row; RUL has exactly 100 values.
- Learning captured: Trailing whitespace can create an apparent extra parsed field and must be removed without altering raw data.
- Decisions made: Use individually downloaded files from the documented public mirror after the official archive transfer repeatedly truncated.
- Blockers: None for ingestion.
- Next action: Implement and test the whitespace parser.

## 2026-09-21 — Module 3 continued

- Goal: Complete raw-data semantic checks and add reproducible validation metadata.
- Changes: Added exact engine-ID validation, cycle-order validation, checksum utilities, raw validation CLI, and `dvc.yaml` stage definition.
- Tests and results: `tests/unit` — 24 passed; raw validation command passed without warnings.
- Learning captured: A count of 100 is insufficient; the validator must also prove the observed IDs are exactly 1–100 and cycles are strictly ordered per engine.
- Decisions made: Keep SHA-256 checksums as a tracked manifest and make the DVC stage validate, not mutate, immutable raw data.
- Blockers: DVC CLI execution has not yet been run in this environment.
- Next action: Inspect DVC availability and complete the Module 3 gate.

## 2026-09-21 — DVC setup

- Goal: Install and initialize DVC for the raw-data validation stage.
- Changes: Installed DVC 3.67.1, initialized `.dvc/` in no-SCM mode, and verified the `validate_raw` stage graph.
- Tests and results: `dvc dag` recognizes `validate_raw`; `dvc repro` is pending because `python` is not available on PATH.
- Learning captured: DVC stage commands execute through the environment’s standard command resolution, so setup must expose the declared Python executable.
- Decisions made: Kept the portable `python -m src.data.validate_raw` stage command; did not hard-code a machine-specific virtualenv path.
- Blockers: Workspace Python is available only as `.venv\\Scripts\\python.exe`.
- Next action: Expose Python on PATH and run the DVC stage.

## 2026-09-21 - Module 4 complete

- Goal: Complete FD001 exploratory analysis and preserve development-only feature decisions.
- Changes: Added lifecycle/settings/sensor EDA notebook content, data dictionary, profile generator, HTML profile report, and profile tests.
- Tests and results: `tests/unit` - 28 passed; `reports/fd001-profile.html` generated successfully.
- Learning captured: Constant sensors create undefined correlation rows; correlated sensors affect linear-model assumptions; association is not physical causality.
- Decisions made: Feature exclusions are based only on development-engine variance at threshold `1e-6`.
- Blockers: None for Module 4.
- Next action: Begin Module 5 labels, deterministic engine manifests, and leakage tests.

## 2026-09-21 - Module 5 complete

- Goal: Construct fixed labels, engine partitions, samples, and leakage controls.
- Changes: Added RUL labels, deterministic manifests, training/validation/final-test sample generation, leakage assertions, materialized manifest artifacts, and `docs/leakage-analysis.md`.
- Tests and results: Full unit suite - 48 passed. Real FD001 evidence: 15,136 training samples, 80 validation samples, and 100 final test samples.
- Learning captured: Engine-level splitting prevents identity leakage; validation cutoffs must use only current and earlier history; labels and lifecycle maxima cannot be feature inputs.
- Decisions made: Keep initial and incremental training as nested subsets of development while asserting only mutually exclusive partition pairs.
- Blockers: None for Module 5.
- Next action: Begin Module 6 shared temporal feature pipeline.

## 2026-09-21 - Module 6 complete

- Goal: Build and verify the shared leakage-safe temporal feature pipeline.
- Changes: Added ordered temporal features, feature-matrix construction, schema metadata, development-only preprocessing, serialized artifact generation, and design documentation.
- Tests and results: Full unit suite - 64 passed; complete development preprocessor fit completed and evidence written to `reports/feature-pipeline-v1.json`.
- Learning captured: Temporal features must be computed from an available-history prefix; variance filtering and scaling are learned only from development data.
- Decisions made: Keep scaling optional because Ridge needs it while tree models do not; preserve one `v1` ordered feature contract for all consumers.
- Blockers: None for Module 6.
- Next action: Begin Module 7 baselines.

## 2026-09-21 - Module 7 metrics and median baseline

- Goal: Establish validated regression metrics and the non-ML median RUL benchmark.
- Changes: Added input-safe RMSE, MAE, and R-squared helpers plus a scikit-learn-compatible median RUL regressor.
- Tests and results: Focused baseline tests - 4 passed; full unit suite - 68 passed.
- Learning captured: The median predictor is the minimum benchmark that every learned regression model must beat; RMSE emphasizes large maintenance-relevant errors while MAE remains directly interpretable in cycles.
- Decisions made: Keep metric validation centralized and fail fast on mismatched, empty, or non-finite inputs.
- Blockers: None.
- Next action: Implement Ridge Regression with standardized development features.

## 2026-09-21 - Module 7 Ridge baseline

- Goal: Add the first learned RUL regression baseline after the median predictor.
- Changes: Added validated `RidgeRULRegressor` using scikit-learn Ridge; it consumes features transformed by the existing development-fitted preprocessor.
- Tests and results: Focused baseline tests - 7 passed; full unit suite - 71 passed.
- Learning captured: Ridge adds an L2 penalty to reduce coefficient magnitude and sensitivity to correlated temporal features; standardization is required so the penalty treats feature coefficients comparably.
- Decisions made: Keep preprocessing separate and reusable instead of duplicating scaling inside the model wrapper; retain clipping at the inference boundary.
- Blockers: None.
- Next action: Implement Logistic Regression and classification metrics for the learning-only failure-within-30 task.

## 2026-09-22 - Module 7 Logistic classification artifact

- Goal: Add the learning-only failure-within-30 classifier and classification evaluation contract.
- Changes: Added validated Logistic Regression, probability and custom-threshold prediction, precision/recall/F1, ROC-AUC, PR-AUC, and confusion-matrix metrics.
- Tests and results: Full unit suite - 75 passed; Ruff checks passed for changed files.
- Learning captured: Accuracy can hide missed failures under class imbalance; PR-AUC and recall-oriented thresholds make the operational trade-off visible.
- Decisions made: Use average precision as PR-AUC, preserve the default 0.5 threshold for comparison, and expose custom thresholds without making classification a served EngineGuard endpoint.
- Blockers: None.
- Next action: Train and benchmark the median, Ridge, and Logistic baselines on the fixed development/validation samples.

## 2026-09-22 - Module 7 complete

- Goal: Benchmark the required Module 7 baselines and record reproducible validation evidence.
- Changes: Added the benchmark runner, JSON/Markdown reports, serialized baseline artifacts, MLflow logging, and declared the MLflow dependency.
- Tests and results: Benchmark completed on 15,136 development samples and 80 fixed validation samples; full unit suite - 75 passed; Ruff passed. Median RMSE 50.2057; Ridge RMSE 20.4961; Logistic PR-AUC 0.9721 at the default threshold.
- Learning captured: The median predictor is the reference floor; Ridge substantially improves the validation error, while classification threshold changes trade precision against recall and do not change ranking metrics such as ROC-AUC or PR-AUC.
- Decisions made: Keep official test data and labels isolated; use standardized development-fitted features for Ridge and Logistic Regression; log local baseline artifacts to the `engineguard-baselines` MLflow experiment.
- Blockers: None.
- Next action: Begin Module 8 Random Forest and XGBoost with GroupKFold by engine.

## 2026-09-22 - Module 8 started

- Goal: Prepare reproducible tree-model training and engine-grouped XGBoost search.
- Changes: Added the XGBoost dependency, Random Forest/XGBoost builders, five-fold `GroupKFold`, the exact 30-configuration search space, and grouped-split contract tests.
- Tests and results: Tree-contract tests - 5 passed; full unit suite - 80 passed; Ruff checks passed for changed tree files; XGBoost 3.4.1 verified.
- Learning captured: GroupKFold prevents rows from the same engine appearing in both a training and validation fold; randomized search must be evaluated by engine, not by individual row.
- Decisions made: Use seed 42, one worker per tree/search process to avoid oversubscription, histogram XGBoost training, and negative RMSE as the search score.
- Blockers: None for the implementation skeleton.
- Next action: Train Random Forest and execute the exact 30-configuration XGBoost search on development samples.

## 2026-09-22 - Module 8 complete

- Goal: Train tree baselines, run the fixed grouped XGBoost search, and compare candidates on fixed validation samples.
- Changes: Added the tree benchmark runner, serialized Random Forest/XGBoost artifacts, frozen `configs/xgboost-candidate-v1.json`, comparison reports, and MLflow logging.
- Tests and results: 30 XGBoost configurations across 5 GroupKFold engine folds completed. Full unit suite - 80 passed; Ruff passed. Ridge RMSE 20.4961, Random Forest RMSE 19.2289, XGBoost RMSE 16.4279 on fixed validation samples. Best grouped-CV XGBoost RMSE was 14.5434.
- Learning captured: Bagging reduces variance through independently fitted trees; boosting sequentially improves residual fit and needs regularization. Grouped folds protect against engine identity leakage.
- Decisions made: Select the candidate using development-only grouped CV, evaluate once on fixed validation samples, use histogram XGBoost with seed 42, and preserve the full 30-trial search evidence.
- Blockers: None.
- Next action: Begin Module 9 final holdout evaluation and explainability; freeze the selected candidate first.

## 2026-09-22 - Module 9 complete

- Goal: Evaluate the frozen XGBoost candidate once on the official FD001 test holdout and produce explainability evidence.
- Changes: Added the protected final evaluator, capped 100-row prediction file, NASA asymmetric score, RUL-band analysis, model card, error-analysis notebook, and global/three-local SHAP artifacts. Logged the final run to `engineguard-final-evaluation`.
- Tests and results: Exactly 100 predictions were produced in range 0-125. Final RMSE 14.2892, MAE 10.4291, R-squared 0.8729, and NASA asymmetric score 377.6640. Full unit suite - 83 passed; Ruff passed.
- Learning captured: Final holdout metrics estimate generalization only after feature selection and model configuration are frozen. SHAP ranks model contributions but does not establish sensor causality.
- Decisions made: Use capped official RUL as the final target, preserve raw predictions for diagnostics, clip only at the prediction boundary, and prevent overwriting final evidence once metrics exist.
- Blockers: None.
- Next action: Begin Module 10 reproducible training and model registry work.

## 2026-09-22 - Module 10 implementation

- Goal: Replace the placeholder training command with a reproducible training and registry stage.
- Changes: Added `src/training/train.py`, a DVC `train` stage, `make train` integration, lineage hashes for raw data/config/DVC/Git, MLflow registration, and candidate/champion aliases.
- Tests and results: Direct training registered `EngineGuardRUL` version 1 with `candidate=1` and `champion=1`; lineage recorded validation RMSE 16.4279; full unit suite - 85 passed; Ruff passed; DVC graph recognized both stages.
- Learning captured: Reproducibility requires recording data checksum, feature schema, code revision, DVC state, configuration, metrics, and artifact identity together.
- Decisions made: Initialize the champion alias only when none exists; preserve an existing champion for later promotion-policy decisions; keep the DVC command portable as `python -m ...`.
- Blockers: `dvc repro train` could not run in the agent shell because `python` is not on the subprocess PATH; the user’s activated `.venv` shell is required for the final DVC gate.
- Next action: Run `dvc repro train` in the activated environment and confirm the lock/output update.

## 2026-09-22 - Module 10 complete

- Goal: Verify the reproducible training and model-registry stage through DVC.
- Changes: User ran `dvc repro train`; the stage completed, `dvc.lock` was updated, and MLflow registered `EngineGuardRUL` version 2.
- Tests and results: DVC completion gate passed; candidate/champion registry workflow is operational. MLflow emitted a non-blocking warning that the logged model has no input signature.
- Learning captured: DVC records stage inputs and outputs while MLflow records model lineage and aliases; both are needed for reproducible promotion workflows.
- Decisions made: Preserve version 2 as the latest registered candidate and keep the signature warning as a follow-up quality improvement rather than changing the frozen evaluation evidence.
- Blockers: None for Module 10.
- Next action: Begin Module 11 FastAPI service implementation.

## 2026-09-22 - Module 11 API implementation

- Goal: Implement the fixed FastAPI serving contract around the champion XGBoost model.
- Changes: Added typed Pydantic request/response schemas, startup-loaded model service, all five endpoints, deterministic risk mapping, request IDs, structured prediction logging, and API contract tests.
- Tests and results: API contract tests - 4 passed; full unit suite - 89 passed; Ruff passed. Health, readiness, model-info, single prediction, batch prediction, and invalid-history behavior are covered.
- Learning captured: Readiness is distinct from process health; loading the model and preprocessor during application startup avoids per-request initialization and makes serving failures explicit.
- Decisions made: Reuse the shared `build_features` and fitted preprocessor for online inference; reject malformed histories through Pydantic before model execution; never log full sensor histories.
- Blockers: Offline/online parity and serving smoke checks remain before Module 11 completion.
- Next action: Add an exact offline/online prediction parity test and run the local serving readiness gate.

## 2026-09-22 - Module 11 complete

- Goal: Verify the FastAPI serving contract and shared offline/online prediction behavior.
- Changes: Added offline/online parity and OpenAPI endpoint smoke tests.
- Tests and results: API contract tests - 6 passed; full unit suite - 91 passed; Ruff passed. All five required endpoints, readiness, bounded output, malformed-history rejection, and model parity are verified.
- Learning captured: Reusing the exact feature builder and fitted preprocessor in the API prevents training/serving skew; readiness confirms model availability while health confirms process availability.
- Decisions made: Keep the champion model loaded once in the application lifespan and use Pydantic validation before feature generation.
- Blockers: None for Module 11.
- Next action: Begin Module 12 container, CI, and coverage work.

## 2026-09-22 - Module 12 implementation

- Goal: Add container, CI, coverage, security-audit, and batch-limit foundations.
- Changes: Added non-root multi-stage `Dockerfile`, `.dockerignore`, GitHub Actions workflow, coverage configuration, `pip-audit` dependency, and the 101-engine batch rejection test. Applied repository Ruff formatting fixes so CI formatting is enforceable.
- Tests and results: Batch contract test - 7 passed; full suite - 93 passed. Measured `src/` coverage is 72%, below the required 80% gate. Docker build is blocked in the agent environment by permission denied on `.docker/buildx/instances`.
- Learning captured: A CI pipeline must prepare reproducible data/model inputs before serving tests; container health and readiness are distinct checks.
- Decisions made: Keep raw data and generated reports outside the Docker context; include the trained model/preprocessor artifacts explicitly in the runtime image; run CI as a non-root container user.
- Blockers: Coverage must reach 80%, and Docker build/smoke verification requires resolving the local Docker daemon permission issue.
- Next action: Add targeted tests for low-covered training/evaluation paths and retry the Docker build from a working Docker context.

## 2026-09-22 - Module 12 complete

- Goal: Verify the test, coverage, Docker, and CI foundations.
- Changes: Added targeted coverage tests for evaluation reports/SHAP artifacts, training data preparation, preprocessing materialization, raw-data quality reporting, and EDA reporting. Added generated coverage and local Docker configuration paths to `.gitignore`. Pinned XGBoost to `3.4.1` to match the serialized model and made the Docker serving image install only runtime dependencies without the unnecessary NCCL dependency.
- Tests and results: Full suite - 101 passed; `src/` coverage - 81.67%; Ruff check and format checks passed. Docker image `engineguard-local:module12` built successfully. Container smoke passed for `/health`, `/ready`, `/model-info`, and a valid 20-cycle `/v1/predict` request; the temporary container was stopped.
- Learning captured: Coverage gates should exercise meaningful pipeline helpers rather than lowering thresholds. A serving image should contain only serving dependencies, while training-only tools such as MLflow, SHAP, and plotting remain outside the runtime image.
- Decisions made: Pin XGBoost to the version used to serialize the champion artifact; keep Docker configuration isolated in the workspace; run the API container as non-root and verify health, readiness, metadata, and prediction behavior.
- Blockers: None for Module 12.
- Next action: Begin Module 13 - Fixed AWS Deployment.
