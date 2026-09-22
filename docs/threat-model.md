# EngineGuard threat model

## Scope

This model covers the public FastAPI service, the Streamlit presentation client,
the serving artifacts, and the training/retraining workflow. It describes the
portfolio deployment boundary; it is not a certification or a complete aviation
safety case.

## Assets

| Asset | Protection goal |
| --- | --- |
| Serving model and feature preprocessor | Integrity and version traceability |
| Sensor histories and prediction responses | Confidentiality and bounded request size |
| API availability | Safe failure and clear readiness state |
| MLflow lineage and aliases | Auditable promotion and rollback |
| Raw FD001 data and labels | Immutability and test-set isolation |
| Deployment secrets | Never committed to Git or container images |

## Threats and controls

| Threat | Example | Current control | Remaining production control |
| --- | --- | --- | --- |
| Oversized request | Memory/CPU exhaustion through a large history | 2 MB body limit, max 2,000 observations, max 100 batch histories | Edge WAF/rate limiting |
| Malformed input | Missing fields, duplicate cycles, non-finite values | Pydantic validation and typed 4xx responses | Centralized abuse monitoring |
| Unauthorized scoring | Untrusted caller uses prediction routes | Optional `ENGINEGUARD_API_KEY` with constant-time comparison | Managed identity or gateway authentication |
| Secret exposure | API key committed to source | `.gitignore`, environment/secrets configuration, no secrets in Dockerfile | Secret manager and rotation |
| Model tampering | Unapproved artifact replaces champion | Versioned artifacts, MLflow aliases, checksums, non-root image | Signed artifacts and restricted registry role |
| Data leakage | Test labels influence tuning | Fixed engine partitions and final-test isolation | Access-controlled training jobs |
| Silent service failure | Model cannot load at startup | `/health`, `/ready`, typed 503 response, startup logging | External alerting and automatic rollback |
| Slow request | Cold start or expensive inference | 10-second request timeout and measured load test | Gateway timeout, autoscaling, rate limits |
| Unsafe promotion | Drift alone causes replacement | Challenger gates, unchanged validation set, audit log, rollback | Human approval for high-impact deployment |

## Authentication behavior

Authentication is disabled when `ENGINEGUARD_API_KEY` is unset, which preserves
the current public demonstration behavior. For a non-public deployment, set the
secret outside source control. Requests to `/v1/predict` and
`/v1/predict/batch` must then include:

```text
X-API-Key: <configured-key>
```

Health and readiness endpoints remain usable for service checks. Invalid
requests are never retried by the API client or service boundary.

## Storage permissions

The current deployment does not provision S3. If AWS storage is added, the
runtime role should be restricted to read-only access to the exact serving
prefix, for example `s3://<bucket>/engineguard/serving/*`, with no bucket-wide
delete, list, or write permissions. Training should use a separate role with
write access to versioned artifacts and experiment outputs only.

## Residual risks

- The public demonstration has no user identity system or rate limiter.
- Vercel cold starts exceed the local p95 latency target in the measured load test.
- FD001 is simulated and does not establish safety for real aircraft.
- API-key authentication is an opt-in boundary for a private deployment, not a
  replacement for a managed identity provider.
