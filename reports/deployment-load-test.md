# Vercel Deployment Load Test

## Run

- Endpoint: `https://nasa-c-mapss.vercel.app/v1/predict`
- Date: 2026-09-22
- Request payload: deterministic valid 20-cycle synthetic history
- Warm-up requests: 5
- Sequential measured requests: 100

## Observed results

| Measure | Result |
| --- | ---: |
| First request latency | 1,038.223 ms |
| Successful requests | 100 / 100 |
| Error rate | 0% |
| Warm p50 | 409.041 ms |
| Warm p95 | 998.333 ms |
| Warm p99 | 1,618.176 ms |
| Warm maximum | 2,573.110 ms |

All responses returned HTTP 200 and the fixed prediction response contract. The
first request is recorded as the cold-start proxy observed by this run; a
separate forced cold-start control is not available through the public endpoint.

## Acceptance assessment

- Prediction success and error-rate goals: passed.
- Latency goal of p95 below 100 ms: not met.
- Interpretation: the public serverless deployment is functionally reliable for
  a portfolio demonstration, but its observed latency is not suitable for a
  strict sub-100-ms interactive requirement. No model or test-set tuning was
  performed in response to this result.

Raw measurements are stored in `reports/deployment-load-test.json` locally.
