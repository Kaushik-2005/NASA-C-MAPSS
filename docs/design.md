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

## Leakage boundaries

Validation and official test rows are not used to fit feature filtering or
scaling. Labels and lifecycle-derived columns are never emitted as features.
