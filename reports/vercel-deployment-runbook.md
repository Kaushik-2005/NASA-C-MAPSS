# Vercel Deployment and Rollback Runbook

## Deploy

1. Confirm the serving model and preprocessor are present under `models/`.
2. Run the unit tests and the Vercel entrypoint test.
3. Push the approved commit to the connected Git repository.
4. Wait for the Vercel deployment to become ready.
5. Verify `/health`, `/ready`, `/model-info`, `/openapi.json`, and one valid
   `/v1/predict` request.

## Roll back

1. Open the project in the Vercel dashboard.
2. Select the last known-good deployment from the Deployments list.
3. Use the deployment actions menu and choose **Promote to Production**.
4. Re-run the endpoint smoke checks above.
5. Record the deployment ID, active model version, reason, and verification
   result in the release notes.

Rollback changes the serving deployment only. It does not retrain a model or
modify the frozen FD001 evaluation artifacts.

## Operational limitation

This project uses Vercel as a serverless portfolio demonstration. The measured
warm p95 latency was 998.333 ms, above the roadmap target of 100 ms. A
production low-latency deployment would require a different hosting boundary
or an explicitly approved paid service.
