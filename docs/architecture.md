# EngineGuard architecture

## System context

Raw FD001 files are validated and versioned, transformed into leakage-safe
temporal features, and used to train and compare the required regression
models. A frozen approved model is later loaded by a typed FastAPI service.

```text
FD001 files -> validation -> labels/splits -> temporal features
             -> train/evaluate -> registry -> API -> monitoring/retraining
```

## Training boundary

Engine identity, time order, and the fixed development/validation manifests
are preserved. No future rows, lifecycle maximum, or target-derived values may
be model inputs. Learned preprocessing is fit on development engines only.

## Inference boundary

The service validates an ordered engine history, calls the same feature
implementation used offline, clips the numeric output to `[0, 125]`, derives
risk, and returns model/schema versions plus a request ID and timestamp.

## Planned component boundaries

- `src/data/`: parsing, schema checks, labels, and deterministic splits.
- `src/features/`: the shared temporal feature contract.
- `src/training/`: reproducible training and registry integration.
- `src/api/`: request validation and inference.
- `src/monitoring/`: quality, drift, latency, and promotion signals.
