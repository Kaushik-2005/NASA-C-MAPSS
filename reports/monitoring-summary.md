# Module 14 Monitoring Summary

The monitoring simulation uses a seed-42 reference sample of 5,000 rows from
the development feature matrix and ten batches of 1,000 validation samples.
Feature computation uses the shared temporal feature builder. Sensor shifts are
applied to raw histories before feature computation.

| Batch | Scenario | Valid | Max PSI | Severity | Result |
| ---: | --- | :---: | ---: | --- | --- |
| 1 | Normal | Yes | 0.190351 | Normal | Scored |
| 2 | Normal | Yes | 0.170062 | Normal | Scored |
| 3 | Normal | Yes | 0.169130 | Normal | Scored |
| 4 | Normal | Yes | 0.137018 | Normal | Scored |
| 5 | Normal | Yes | 0.154273 | Normal | Scored |
| 6 | Sensor shift | Yes | 10.983537 | Critical | Scored with alert |
| 7 | Sensor shift | Yes | 10.887861 | Critical | Scored with alert |
| 8 | Sensor shift | Yes | 10.925971 | Critical | Scored with alert |
| 9 | Missing `sensor_5` | No | — | — | Quarantined; no prediction |
| 10 | Normal recovery | Yes | 0.144593 | Normal | Scored |

The drift signal is evidence of a distribution change, not proof that model
accuracy declined or that retraining should be promoted automatically. Batch 9
is rejected at schema validation, and batch 10 demonstrates safe recovery.

## Performance and service alerts

The frozen validation RMSE is `16.4278823` cycles. The performance warning
threshold is `19.7134588` cycles (`120%` of the frozen value). The ten simulated
valid batches produced RMSE values from `14.861951` to `16.133113`, so none
triggered a performance warning.

The pipeline accepts observed service p95 latency per batch through
`service_p95_by_batch`. It raises a warning only after two consecutive valid
batches exceed `100 ms`; the rule is covered by deterministic unit tests. The
offline feature simulation does not fabricate HTTP latency measurements, so
those values are `null` in this run.

Machine-readable per-batch reports, the alert log, and Evidently HTML/JSON
comparison reports are under `reports/monitoring/` and
`reports/monitoring/evidently/`.
