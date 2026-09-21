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
