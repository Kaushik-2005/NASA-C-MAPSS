# Champion–Challenger Retraining Summary

The fixed Module 15 simulation used development engines only and evaluated both
models on the unchanged 20-engine validation partition (80 samples).

| Model | Training engines | Validation RMSE | Validation MAE | Single-row p95 |
| --- | --- | ---: | ---: | ---: |
| Initial champion | Initial-training partition | 16.9502 | 12.8646 | — |
| Challenger | All development engines | 16.4279 | 12.3607 | 18.658 ms |

The challenger improved RMSE by approximately 3.08%, improved MAE, produced 80
bounded predictions, stayed below the 100 ms p95 latency gate, and had complete
local artifact lineage. The promotion policy therefore returned `promote=true`.

This simulation records a promotion decision only. It does not replace the
deployed Vercel model or alter the official final-test artifacts.

The transition audit registered the initial champion as MLflow version 2 and
the challenger as version 3. After all gates passed, both `candidate` and
`champion` point to version 3. The promotion evidence includes a passing
114-test unit-suite run; the previous champion version and local champion
artifact remain available for rollback.

Machine-readable metrics and the transition audit are under
`reports/retraining/`.
