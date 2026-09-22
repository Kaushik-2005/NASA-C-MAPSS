# EngineGuard cost analysis

## Decision

Keep the portfolio demonstration on Vercel for the API and run the Streamlit UI
locally or on a free Streamlit Community Cloud deployment. Do not provision AWS
resources for this project without a separate approval and budget decision.

## Demonstration options

| Option | Cost posture | Operational trade-off |
| --- | --- | --- |
| Local FastAPI + local Streamlit | No hosted compute cost | Only available from the developer machine |
| Vercel API + local Streamlit | No dedicated always-on server | Serverless cold starts; measured public p95 was 998.333 ms |
| Vercel API + Streamlit Community Cloud | No dedicated paid compute assumed | Two deployments and an external API dependency |
| AWS ECS/Fargate + ECR + S3 + CloudWatch | Paid usage and account setup required | More control, but more infrastructure and operational work |

## AWS planning estimate

The following is a planning model, not a billing quote. Actual cost depends on
region, CPU/memory size, request volume, storage, log retention, data transfer,
free-tier eligibility, and whether a service runs continuously.

| AWS component | Always-on pattern | Scheduled/batch pattern |
| --- | --- | --- |
| ECS/Fargate | Continuous hourly vCPU and memory charges | Pay only while scoring or retraining jobs run |
| ECR | Small image-storage charge; usually low for one image | Same storage, fewer image versions retained |
| S3 | Low storage/request cost for two small model artifacts | Same storage; lifecycle rules can remove old artifacts |
| CloudWatch | Log ingestion and retention charges grow with request volume | Lower volume if batch jobs emit summarized logs |
| Networking | Possible load balancer and data-transfer charges | Avoid an always-on load balancer for scheduled jobs |

For this small FD001 demonstration, an always-on ECS service would pay for
capacity even when nobody is using it. A scheduled batch job is more
cost-efficient for offline scoring, while serverless Vercel is the simpler
public demonstration boundary. A production AWS estimate should be generated
with the [AWS Pricing Calculator](https://calculator.aws/) after selecting a
region and resource sizes.

## Cost controls

- Keep raw data and reports outside the serving image.
- Retain only approved serving artifacts in the deployment bundle.
- Use S3 prefixes and lifecycle policies for versioned artifacts.
- Set CloudWatch log retention instead of retaining logs indefinitely.
- Prefer scheduled batch inference when real-time scoring is unnecessary.
- Add request limits and timeouts before exposing a paid service.
- Delete demonstration resources after an approved experiment.

## Current project decision

No AWS resources are required to reproduce the project. The documented AWS
architecture remains a future production option, while the measured Vercel
deployment provides the current public API demonstration without provisioning
paid infrastructure.
