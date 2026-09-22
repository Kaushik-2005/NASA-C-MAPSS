# Learning Notes

## Module 2 — ML Foundations Using FD001

### Concepts

RUL is a regression target measured in cycles. `failure_within_30` is a binary
learning target. The NumPy implementations expose the optimization steps before
the project uses library models.

### Formulas

Linear regression minimizes mean squared error. Ridge adds an L2 penalty to the
feature weights. Logistic regression uses the sigmoid function and binary
cross-entropy (log loss).

### EngineGuard application

The FD001 verification uses development engines only and standardizes `cycle`
and `sensor_2`. The NumPy and scikit-learn results are compared on the same
data. Learning-rate and L2 experiments show how convergence and regularization
change training behavior.

### Trade-offs and common mistakes

Gradient descent is easy to inspect but depends on feature scale and learning
rate. Ridge can reduce coefficient variance with correlated sensors but may
underfit nonlinear degradation. A row-level random split leaks engine-specific
patterns, so EngineGuard splits by engine ID.

### Evidence

Run `python -m scripts.verify_foundations_fd001`. The recorded artifact is
`reports/foundations-fd001.json`; it covers 80 development engines and 16,656
rows. The script is called by the final notebook cell.

### Interview questions

1. Why is row-level random splitting invalid? It can place the same engine in
   both partitions and make validation look better than it is.
2. Why standardize features for Ridge? The penalty acts on coefficient size, so
   different feature scales would receive unequal effective regularization.
3. Why is accuracy not enough for failure classification? A majority-class
   model can have good accuracy while missing near-term failures.

## Module 3 — FD001 Ingestion, Validation, and Versioning

### Concepts

Ingestion converts the whitespace-separated source files into a typed schema.
Semantic validation checks engine counts, cycle ordering, field counts, and RUL
length rather than only checking data types.

### EngineGuard application

The three raw FD001 files remain immutable. Checksums are stored in
`data/checksums.sha256`. Validation writes `reports/fd001-data-quality.json`
and the DVC `validate_raw` stage stops before downstream work if a contract
fails.

### Common mistakes

Do not silently repair duplicate cycles, missing engines, or shifted columns.
Do not use official test labels while selecting features or tuning models.

### Evidence

`python -m src.data.validate_raw --report reports/fd001-data-quality.json`
passes, and `dvc repro validate_raw` completes with the workspace-local DVC
cache configuration documented in the session log.

### Interview questions

1. Why keep raw files immutable? It preserves provenance and makes later
   results reproducible.
2. Why validate cycle ordering? Temporal features and labels depend on a valid
   sequence; duplicate or missing cycles can change their meaning.
3. Why use DVC here? It records data-stage dependencies and outputs alongside
   Git without placing large raw files in the Git history.

## Module 16 — Reliability, Security, and Cost

### Concepts

The API separates invalid-input failures, request-size limits, authentication
failures, readiness failures, and inference failures. Invalid requests are
rejected without retries. Optional API-key authentication protects prediction
routes when a private deployment needs it.

### EngineGuard application

Prediction batches are capped at 100 engines, histories are bounded to 2,000
cycles and at least 20 cycles, request bodies are limited to 2 MB, and inference
has a timeout. `X-API-Key` is checked only when `ENGINEGUARD_API_KEY` is set;
health, readiness, and model metadata remain available for service checks.

### Trade-offs and common mistakes

Keeping the public Vercel demo unauthenticated makes it easy to inspect, but it
also leaves it open to abuse. A private deployment should set the API key via
the platform secret store. S3 permissions should be limited to the exact model
prefix and read operations required by the serving process.

### Evidence

See `docs/threat-model.md`, `docs/security-checklist.md`, and
`docs/cost-analysis.md`. `scripts/security_audit.py` checks high-confidence
secret patterns and the expected model artifact allowlist.

### Interview questions

1. Why are typed errors useful? Clients can distinguish bad input, auth failure,
   overload, and server failure without parsing free-form messages.
2. Why is drift not enough to promote a model? Distribution change does not
   prove that a challenger improves labeled prediction quality.
3. Why compare always-on and scheduled inference? Scheduled work can reduce
   idle cost when predictions are needed in batches rather than continuously.
