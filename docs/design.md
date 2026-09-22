# EngineGuard design

## Temporal feature contract

The shared feature builder in `src/features/build_features.py` receives only
the history available up to a prediction cycle. It produces one ordered row
under schema version `v1` with current values, rolling statistics, a five-cycle
delta, a twenty-cycle slope, current cycle, and capped history count.

`build_feature_matrix` applies the same function to each labeled sample and
never includes rows after that sample's cycle. The API and future batch scorer
must call this same builder rather than reimplementing feature logic.

## Preprocessing lifecycle

`FeaturePreprocessor` fits variance filtering on development features only.
Optional scaling is fit after filtering and is enabled only for models that
require it. The ordered input schema is checked at transform time, and the
fitted object is serialized with Joblib together with the `v1` feature metadata.

The current development-only artifact is described in
`reports/feature-pipeline-v1.json`. It must be regenerated when feature order,
exclusions, or preprocessing parameters change.

## Deployment boundary

The local and containerized FastAPI service is also deployable as a Vercel
Python Function through `api/index.py`. Vercel loads the same `src.api.app:app`
instance and routes all requests through the existing FastAPI contract. Only
the candidate model and development preprocessor are deployment artifacts;
raw data, reports, training stores, and experimental models remain excluded.

Vercel is a serverless demonstration boundary: model loading may occur during
cold starts, and training, MLflow, DVC, monitoring reports, and retraining stay
outside the request function. The Docker image remains the reproducible local
serving artifact.

## Reliability and security boundary

The API limits batch requests to 100 engines, history to 2,000 cycles, and
request bodies to 2 MB. Histories must contain at least 20 strictly increasing
cycles. Invalid input returns a typed client error and is not retried. A
configured `ENGINEGUARD_API_KEY` protects `/v1/*` routes through the
`X-API-Key` header; health, readiness, and model metadata remain available for
service checks. The public Vercel demonstration leaves this optional key unset.

`docs/threat-model.md` records the trust boundaries and residual risks.
`scripts/security_audit.py` checks for high-confidence secret patterns and
unexpected model artifacts. If model storage is moved to S3, the serving role
must receive read access only to the required versioned model prefix.

## Demonstration UI boundary

The optional Streamlit application under `streamlit_app/` is a presentation
client, not a second inference service. Its prediction page sends requests to
the deployed Vercel API, so validation, feature generation, model versioning,
and risk mapping remain centralized. Curated, non-sensitive EDA summaries are
bundled with the UI; raw data, MLflow stores, and training artifacts are not.

## Leakage boundaries

Validation and official test rows are not used to fit feature filtering or
scaling. Labels and lifecycle-derived columns are never emitted as features.

## Retraining and promotion flow

Module 15 simulates newly labeled incremental-training engines without using
official test labels. The initial champion is trained on the fixed
initial-training partition; the challenger uses all development engines. Both
are evaluated on the unchanged validation-engine samples. Retraining is
triggered by two consecutive valid batches with feature PSI at or above 0.20,
or labeled RMSE above 120% of the frozen champion validation RMSE.

Promotion requires every gate in `src/training/retrain.py`: at least 3% RMSE
improvement, no worse MAE, complete bounded predictions, p95 latency below
100 ms, registered lineage/artifacts, and a passing unit suite. MLflow stores
both runs, assigns the challenger the `candidate` alias, and moves
`champion` only after the decision passes. A failed challenger retains the
champion; the transition audit records the rejection reasons and the previous
champion version for rollback.
