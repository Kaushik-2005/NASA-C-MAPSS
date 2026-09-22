# Technical Decisions

## ADR-001: Preserve fixed FD001 contracts in typed project configuration

- Status: Accepted
- Context: The roadmap freezes dataset, target cap, minimum history, seed, and feature schema values.
- Decision: Expose these values through a small Pydantic configuration model and keep runtime commands separate from future pipeline implementations.
- Alternatives considered: Ad-hoc constants in each module, which risks contract drift; a full settings service, which is unnecessary for the local foundation.
- Consequences: Later modules have a single source for core constants and can validate configuration at startup.
- Evidence: `src/config.py`, `tests/test_smoke.py`

## ADR-002: Use Vercel Functions for the no-cost public API demonstration

- Status: Accepted
- Context: AWS deployment would require paid-account setup and the user does not want to provision paid infrastructure. Hugging Face Docker Spaces also require paid compute, while the existing FastAPI service is small enough for a serverless demonstration.
- Decision: Export the existing FastAPI app through `api/index.py` and deploy it as a Vercel Python Function. Keep Docker as the local reproducible serving artifact and retain AWS only as a future production option.
- Alternatives considered: AWS ECS/Fargate and S3, which add paid infrastructure; Hugging Face Docker Spaces, which require paid compute; Render Free, which preserves Docker but may suspend inactive services.
- Consequences: The public demo must account for cold starts, function bundle limits, and serverless execution. Training and monitoring remain outside the request function.
- Evidence: `api/index.py`, `vercel.json`, `docs/deployment-vercel.md`, and the Vercel FastAPI documentation.

## ADR-003: Gate challenger promotion with fixed validation evidence

- Status: Accepted
- Context: Retraining must respond to drift or delayed-label degradation without allowing a candidate to replace a working model based on drift alone.
- Decision: Train the initial champion and challenger on the fixed FD001 partitions, compare them on unchanged validation samples, require every promotion gate, register both MLflow runs, and move the `champion` alias only after the gates pass.
- Alternatives considered: Automatic promotion on PSI alone, which can promote a model when drift does not imply accuracy improvement; manual file replacement without registry lineage, which weakens rollback and auditability.
- Consequences: Promotion is deliberately conservative and local to the registry simulation. The public Vercel artifact is not changed automatically. A prior model version and artifact backup remain available for rollback.
- Evidence: `src/training/retrain.py`, `reports/retraining/transition-audit.json`, and `tests/unit/test_retrain.py`.

## ADR-004: Use Streamlit as a separate presentation client

- Status: Accepted
- Context: The project needs a simple portfolio interface for model testing, EDA, and MLOps explanations while preserving one production prediction path.
- Decision: Deploy a lightweight Streamlit app separately from Vercel. The app calls the public FastAPI endpoints for predictions and bundles only curated EDA summaries and explanatory content.
- Alternatives considered: Loading a second model inside Streamlit, which can drift from the deployed model; adding UI routes to FastAPI, which couples presentation concerns to the serving contract.
- Consequences: The UI requires a separate deployment and must handle API availability errors. It remains cheap to operate and can be updated independently without changing the model service.
- Evidence: `streamlit_app/app.py`, `streamlit_app/api_client.py`, and `docs/design.md`.

## ADR-005: Make API-key authentication optional by deployment

- Status: Accepted
- Context: The public Vercel demonstration should remain easy to inspect, while a private deployment needs a simple access control boundary.
- Decision: Read `ENGINEGUARD_API_KEY` from the deployment secret store. When configured, require it in `X-API-Key` for prediction routes; otherwise preserve the public demo behavior. Keep health, readiness, and model metadata available for service checks.
- Alternatives considered: Hard-code a key, which would expose a secret; add a full identity provider, which is unnecessary for this project; protect only the UI, which would not protect the API itself.
- Consequences: Private deployments must configure one secret and clients must send one header. This is an access-control layer, not a complete abuse-prevention system.
- Evidence: `src/api/security.py`, `src/api/app.py`, `tests/unit/test_api_contract.py`, and `docs/threat-model.md`.

## ADR-006: Keep cost analysis at the deployment-pattern level

- Status: Accepted
- Context: The user does not want to provision paid AWS infrastructure for the demonstration.
- Decision: Document local, Vercel, Streamlit Community Cloud, always-on AWS, and scheduled batch patterns without creating cloud resources. Use the AWS Pricing Calculator for any future estimate and require explicit approval before provisioning.
- Alternatives considered: Creating a live AWS stack to obtain an exact bill, which adds cost and credentials risk; omitting cost analysis, which would leave the deployment trade-off unexplained.
- Consequences: The cost document contains planning comparisons rather than an invoice-level estimate. The current public API remains on Vercel and UI deployment is optional.
- Evidence: `docs/cost-analysis.md`, `docs/security-checklist.md`, and `docs/threat-model.md`.
