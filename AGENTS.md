# AGENTS.md — EngineGuard Development Instructions

## Purpose

This repository builds **EngineGuard**, a production-oriented remaining-useful-life prediction platform based on the NASA C-MAPSS FD001 dataset.

Use `roadmap.md` as the authoritative project specification. This file defines how an AI coding agent must execute that roadmap, protect the fixed ML contracts, teach the underlying concepts, verify work, and preserve project history.

The goal is not merely to produce working code. The user must finish the project able to explain:

- The ML theory behind every model and metric.
- The data and leakage controls.
- The system architecture and trade-offs.
- The MLOps lifecycle from raw data to monitoring and rollback.
- The evidence supporting every resume claim.

---

## 1. Instruction Priority

Follow instructions in this order:

1. The user's current explicit request.
2. This `AGENTS.md`.
3. The fixed project contract and module requirements in `roadmap.md`.
4. Existing design decisions in `docs/decisions.md`.
5. Existing repository conventions.

If instructions conflict, stop and explain the conflict before changing code or project contracts.

Do not silently reinterpret the roadmap to make implementation easier.

---

## 2. Source-of-Truth Files

Treat these files as durable project memory:

| File | Purpose |
| --- | --- |
| `roadmap.md` | Fixed project contract, module order, deliverables, and completion gates |
| `AGENTS.md` | Rules for executing the roadmap |
| `docs/tracker.md` | Current module, task status, evidence, blockers, and next action |
| `docs/learning.md` | Concise learning notes, formulas, examples, and interview explanations |
| `docs/design.md` | Current system architecture, component contracts, and data flows |
| `docs/decisions.md` | Important technical decisions and their trade-offs |
| `docs/session-log.md` | Chronological record of work completed and verified |

Create a missing tracking file when the first relevant module needs it. Do not create duplicate files with numbered or alternate names.

Read `roadmap.md`, `docs/tracker.md`, `docs/design.md`, and `docs/decisions.md` before starting or resuming implementation.

---

## 3. Non-Negotiable Project Contract

The following values are frozen unless the user explicitly approves a roadmap change:

### Dataset

- Use NASA C-MAPSS **FD001 only** for the core project.
- Required files:
  - `train_FD001.txt`
  - `test_FD001.txt`
  - `RUL_FD001.txt`
- Expected schema: `unit_id`, `cycle`, 3 operational settings, and 21 sensors.
- Raw source data must remain immutable.

### Target

```text
raw_rul = maximum_training_cycle_for_engine - current_cycle
target_rul = min(raw_rul, 125)
prediction = clip(model_output, 0, 125)
```

Never include maximum lifecycle length, `raw_rul`, `target_rul`, future rows, or final failure information as input features.

### Risk mapping

```text
0–30 cycles   -> critical
31–60 cycles -> warning
61–125 cycles -> healthy
```

Risk is deterministic application logic, not a separately deployed classifier.

### Engine partitions

```text
validation: unit_id % 5 == 0
development: unit_id % 5 != 0

initial training: unit_id <= 75 and unit_id % 5 != 0
incremental training: unit_id > 75 and unit_id % 5 != 0
```

Never use a row-level random split.

### Samples

- Require at least 20 history cycles.
- Generate training samples from cycle 20 onward.
- Generate validation samples at 60%, 70%, 80%, and 90% of engine lifetime.
- Generate one final prediction for each of the 100 official test engines.

### Models

Required regression models:

1. Median constant predictor.
2. Ridge Regression.
3. Random Forest Regressor.
4. XGBoost Regressor.

Required learning-only classifier:

1. Logistic Regression for `failure_within_30`.

Do not replace the classical ML pipeline with deep learning. Deep-learning experiments begin only after the core definition of done is satisfied.

### Reproducibility

Use random seed `42` wherever randomness is involved.

---

## 4. Test-Set Isolation Policy

The official FD001 test set is a final holdout.

Before Module 9:

- It may be parsed and schema-validated.
- Feature-generation code may be tested structurally against it.
- `RUL_FD001.txt` must not influence feature selection, preprocessing, model choice, hyperparameters, thresholds, or acceptance criteria.
- Do not report test RMSE or MAE.

In Module 9:

- Freeze features, preprocessing, model family, and hyperparameters first.
- Run the official test evaluation once for the final report.
- Record the Git commit, data checksum, configuration, and model version used.

After Module 9:

- Do not tune against test results.
- If a quality target is missed, document the gap and error analysis.
- Any later model change requires a new clearly versioned experiment and must not overwrite the original final-test evidence.

Treat any accidental test-label use during development as a leakage incident. Record it in `docs/session-log.md`, invalidate affected results, and rebuild from a clean boundary.

---

## 5. Module Execution Protocol

Work on one roadmap module at a time unless the user explicitly requests otherwise.

### Before starting a module

1. Read the module's `Learn`, `Build`, `Deliver`, and `Done when` sections.
2. Inspect the current repository rather than assuming files are absent.
3. Read the relevant source-of-truth documents.
4. Check `git status` and preserve unrelated user changes.
5. Update `docs/tracker.md` to mark the module `IN PROGRESS`.
6. List the exact tasks and verification commands for the module.
7. Identify prerequisites, missing data, credentials, or user decisions.

### While implementing

1. Make small, coherent changes.
2. Keep notebooks exploratory; move reusable logic into `src/`.
3. Add or update tests with each behavior change.
4. Run the narrowest relevant test first.
5. Run broader regression tests before declaring completion.
6. Update learning and design documents as knowledge or architecture changes.
7. Record material trade-offs in `docs/decisions.md`.
8. Use actual artifacts and measurements; never invent metrics.

### Before completing a module

1. Verify every deliverable exists.
2. Run every completion-gate check.
3. Record commands and observed results.
4. Confirm no fixed contract was changed.
5. Update `docs/tracker.md` with evidence paths.
6. Update `docs/session-log.md`.
7. Mark the module `COMPLETE` only when its `Done when` condition is satisfied.

If code is implemented but the gate is not verified, use `IMPLEMENTED — VERIFICATION PENDING`, not `COMPLETE`.

---

## 6. Tracker Format

Maintain `docs/tracker.md` with this structure:

```markdown
# Project Tracker

## Current Module

- Module: 3 — FD001 Ingestion, Validation, and Versioning
- Status: IN PROGRESS
- Started: YYYY-MM-DD
- Last updated: YYYY-MM-DD
- Current task: Implement Pandera schema
- Next action: Add invalid-cycle fixtures
- Blockers: None

## Module Status

| Module | Status | Evidence | Notes |
| --- | --- | --- | --- |
| 1 | COMPLETE | `pyproject.toml`, test output | Gate verified |
| 2 | COMPLETE | notebook, tests | Gate verified |
| 3 | IN PROGRESS | `src/data/` | Validation tests pending |
| 4 | NOT STARTED | — | — |

## Current Module Checklist

- [x] Parse raw data
- [x] Assign schema
- [ ] Validate engine count
- [ ] Add failure fixtures
- [ ] Run completion gate

## Verification Evidence

| Command | Result | Date |
| --- | --- | --- |
| `pytest tests/unit/test_data_schema.py -q` | 8 passed | YYYY-MM-DD |

## Blockers and Risks

- None.
```

Allowed status values:

- `NOT STARTED`
- `IN PROGRESS`
- `IMPLEMENTED — VERIFICATION PENDING`
- `BLOCKED`
- `COMPLETE`

Never mark a module complete based only on file existence.

---

## 7. Learning Workflow

This is a learning project, not only a code-generation exercise.

For each module:

1. Explain the core concepts before or alongside implementation.
2. Connect each concept directly to FD001 and EngineGuard.
3. Explain why the selected method is appropriate and what alternatives exist.
4. Record concise notes in `docs/learning.md`.
5. Add at least three interview questions with short answers.
6. End the module with a knowledge check or ask the user to explain one key trade-off.

For algorithms, learning notes should include where applicable:

- Objective or loss function.
- Important assumptions.
- Main hyperparameters.
- Computational trade-offs.
- Failure modes.
- Why the method is or is not suitable for EngineGuard.

Do not hide important reasoning behind library calls. For the foundations module, preserve the NumPy implementations even after using scikit-learn.

---

## 8. Documentation Rules

### `docs/learning.md`

Organize notes by roadmap module:

```markdown
## Module 8 — Random Forest and XGBoost

### Concepts
### Formulas
### EngineGuard application
### Trade-offs
### Common mistakes
### Interview questions
```

Write explanations in clear language suitable for later interview revision.

### `docs/design.md`

Keep the current design, not a chronological diary. Include:

- System context.
- Training pipeline.
- Inference pipeline.
- Data and feature contracts.
- Model lifecycle.
- Monitoring and retraining flows.
- Failure behavior.
- Security and deployment boundaries.

Update existing sections instead of appending conflicting versions.

### `docs/decisions.md`

Record only decisions with meaningful alternatives or long-term consequences:

```markdown
## ADR-003: Use engine-level deterministic partitions

- Status: Accepted
- Context:
- Decision:
- Alternatives considered:
- Consequences:
- Evidence:
```

Do not create ADRs for trivial syntax or formatting choices.

### `docs/session-log.md`

Append one concise entry per work session:

```markdown
## YYYY-MM-DD — Module 6

- Goal:
- Changes:
- Tests and results:
- Learning captured:
- Decisions made:
- Blockers:
- Next action:
```

---

## 9. Data and Artifact Discipline

- Never modify files under `data/raw/`.
- Record raw-file checksums.
- Keep raw, interim, and processed data separate.
- Track data and reproducible pipeline outputs with DVC.
- Do not commit large datasets, model binaries, MLflow stores, generated reports, credentials, or temporary artifacts directly to Git.
- Add generated and sensitive files to `.gitignore` or `.dvcignore` as appropriate.
- Preserve ordered feature names with every model.
- Store feature schema version, data version, code version, and model version together.
- Treat a feature-order change as a contract change.
- Do not overwrite final metrics or model artifacts without versioning them.

If the FD001 source is downloaded from a mirror rather than the original source, document the source and verify file structure and checksums before continuing.

---

## 10. ML Implementation Standards

- Fit scalers, variance filters, feature selectors, and models using development data only.
- Put learned transformations inside serializable pipelines where possible.
- Use group-aware validation by `unit_id`.
- Use the exact sample and feature policies in `roadmap.md`.
- Compare every candidate with the median and Ridge baselines.
- Report RMSE and MAE together.
- Measure training time, inference time, and serialized model size.
- Clip predictions only at the defined inference boundary; retain raw predictions for diagnostic analysis.
- Analyse underprediction and overprediction separately because their operational risks differ.
- Treat SHAP as an explanation of model behavior, not proof of physical causality.
- Do not claim a model is production-ready solely because offline RMSE is good.

When training is expensive, first use a documented smoke subset. Never report smoke-run metrics as final metrics.

---

## 11. Software Engineering Standards

- Support the Python version declared in `pyproject.toml`.
- Use type hints for public functions and interfaces.
- Prefer small modules with explicit boundaries.
- Use Pydantic models for API contracts.
- Use structured logging and never use `print` for production observability.
- Raise specific exceptions with actionable messages.
- Keep configuration outside business logic.
- Do not duplicate feature engineering between training and serving.
- Avoid global mutable state except controlled application lifecycle resources.
- Load the production model once during API startup.
- Include model and schema versions in prediction outputs and logs.
- Never log full sensor histories by default.

Add docstrings when behavior, invariants, or units are not obvious. Avoid comments that merely restate code.

---

## 12. Testing Standards

Use tests to protect contracts, not merely raise coverage.

Required categories:

- Unit tests for labels, splits, features, metrics, and risk mapping.
- Schema tests for valid and invalid raw data.
- Leakage tests for engine partitions and temporal features.
- Contract tests for every API endpoint.
- Offline/online feature-parity tests.
- Model serialization and prediction-invariant tests.
- Integration tests for the training pipeline.
- End-to-end smoke tests for training and serving.
- Regression tests for frozen small fixtures.
- Monitoring and promotion-policy tests.

Tests must be deterministic and must not require full model training unless explicitly marked.

Do not weaken or delete a valid test merely to make CI pass. Fix the behavior or explicitly update the contract with user approval.

---

## 13. Evaluation and Claims

The acceptance thresholds in `roadmap.md` are targets, not guaranteed facts.

- Never fabricate, round aggressively, or selectively report results.
- Never state that a target was achieved before running the relevant evaluation.
- Separate validation results from official test results.
- State the number of engines and samples behind every reported metric.
- Preserve failed experiments when they teach a useful lesson.
- Include limitations and negative results in the final report.
- Use measured values only in README badges, portfolio text, and resume bullets.

Any claim such as “improved RMSE by 20%” must identify the baseline, dataset partition, metric implementation, and artifact containing the result.

---

## 14. API and Monitoring Contracts

Do not change the five required endpoints or response fields without explicit approval.

Required endpoints:

- `POST /v1/predict`
- `POST /v1/predict/batch`
- `GET /health`
- `GET /ready`
- `GET /model-info`

Required prediction response fields:

- `unit_id`
- `predicted_rul`
- `risk_level`
- `model_version`
- `feature_schema_version`
- `request_id`
- `predicted_at`

Monitoring must distinguish:

- Infrastructure and service failures.
- Schema and data-quality failures.
- Feature or prediction drift.
- Labeled performance degradation.
- Retraining triggers.
- Promotion decisions.

Drift is evidence of distribution change, not proof that model accuracy declined. Never auto-promote a model solely because drift was detected.

---

## 15. Cloud and Destructive Actions

Local development, tests, Docker builds, and configuration preparation are authorized by normal project work.

Before creating paid or externally visible AWS resources:

1. Present the resources to be created.
2. Estimate likely demonstration cost.
3. Explain how they will be stopped or deleted.
4. Obtain explicit user approval.

Never:

- Provision AWS resources merely to test credentials.
- Store credentials in the repository.
- Broaden IAM permissions to bypass an access error.
- Delete buckets, registries, logs, experiments, models, or cloud resources without confirming the exact target and scope.
- Push, merge, tag, publish, or deploy unless the user requested or approved it.

Prefer dry runs, local emulators, and configuration validation before external changes.

---

## 16. Git and Change Safety

- Inspect `git status` before editing.
- Treat existing modifications as user-owned.
- Do not overwrite unrelated changes.
- Do not use destructive Git commands.
- Do not amend, rebase, force-push, or rewrite history without explicit instruction.
- Do not commit or push unless the user asks.
- Keep generated artifacts out of source control.
- Make each change reviewable and explain unexpected diffs.

If a requested change conflicts with uncommitted user work, stop and ask rather than guessing.

---

## 17. Handling Ambiguity and Blockers

First inspect the roadmap, tracking files, code, configuration, tests, and logs. Do not ask the user for information already present in the repository.

Ask the user only when:

- A choice would change the frozen project contract.
- Credentials or external authorization are required.
- A destructive operation is necessary.
- Multiple plausible choices have materially different outcomes.
- The roadmap and existing implementation irreconcilably conflict.

When blocked, update `docs/tracker.md` with:

- Exact blocker.
- Evidence or error message.
- What was attempted.
- Safe next action.
- Whether user input is required.

Continue with independent in-scope work when possible.

---

## 18. Agent Communication Format

At the beginning of a work session, report:

```text
Current module:
Goal for this session:
Files expected to change:
Verification planned:
```

During work, provide concise updates after meaningful milestones or when a blocker appears.

At the end of a work session, report:

```text
Completed:
Verified:
Learning captured:
Design/decision updates:
Remaining:
Next action:
```

Do not claim success from implementation alone. Lead with verified outcomes.

---

## 19. Prohibited Shortcuts

Do not:

- Replace FD001 with another dataset for the core roadmap.
- Randomly split rows from the same engine.
- Use official test labels during development.
- Tune hyperparameters after viewing final test results.
- Compute features from future cycles.
- Train on labels or lifecycle-derived leakage columns.
- Duplicate preprocessing in the API.
- Skip baselines and report only XGBoost.
- Add deep learning before completing the core pipeline.
- Mark modules complete without gate evidence.
- Invent performance, latency, coverage, cost, or deployment claims.
- Treat drift as proof of concept drift.
- Promote a challenger that fails any promotion gate.
- Provision paid infrastructure without approval.
- Rewrite the roadmap contract without explicit user direction.

---

## 20. Project Completion

Before declaring EngineGuard complete:

1. Verify every module in `roadmap.md` is marked `COMPLETE` with evidence.
2. Run the full automated test suite.
3. Run the reproducibility check.
4. Confirm exactly 100 official test predictions exist.
5. Verify all fixed API contracts.
6. Verify the deterministic drift scenario and recovery.
7. Verify challenger promotion or rejection and champion rollback.
8. Confirm all tracked claims come from measured artifacts.
9. Complete README, data card, model card, benchmark, cost analysis, and threat model.
10. Record unresolved limitations honestly.

The final handoff must state:

- What was built.
- What was verified.
- Which quality goals were met or missed.
- Known limitations.
- Exact reproduction commands.
- Safe next improvements that do not alter the core evidence.
