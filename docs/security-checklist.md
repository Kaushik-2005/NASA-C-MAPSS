# Security checklist

- [x] Request body capped at 2 MB.
- [x] Batch requests capped at 100 engine histories.
- [x] Individual histories capped at 2,000 observations.
- [x] Minimum history and strict cycle ordering validated.
- [x] Non-finite sensor values rejected.
- [x] Typed validation, authentication, timeout, and readiness errors returned.
- [x] Optional API-key authentication available for private deployments.
- [x] Invalid requests are not retried.
- [x] Secrets and credentials are excluded from Git and Docker inputs.
- [x] Docker runtime uses a non-root user.
- [x] Serving artifacts are explicitly allowlisted.
- [x] S3 least-privilege policy documented before any AWS provisioning.
- [x] Threats and residual risks documented.
- [x] Cost trade-offs documented before paid infrastructure.

The public Vercel demonstration intentionally keeps authentication disabled.
Set `ENGINEGUARD_API_KEY` in a private deployment to activate the API-key
boundary for prediction routes.
