# EngineGuard problem definition

EngineGuard predicts a turbofan engine's remaining useful life (RUL), measured
in cycles, from its observed sensor history. The core dataset is NASA C-MAPSS
FD001: 100 training engines and 100 held-out test engines, with one operating
condition and one fault mode.

The supervised target is capped RUL:

```text
raw_rul = maximum_training_cycle_for_engine - current_cycle
target_rul = min(raw_rul, 125)
```

Predictions are clipped to `[0, 125]`. Application risk is deterministic:
`0–30` is critical, `31–60` is warning, and `61–125` is healthy. Risk is not a
separately trained classifier.

Validation is engine-level and deterministic (`unit_id % 5 == 0`), while the
remaining engines form development data. The official test set remains a final
holdout until Module 9.

This is a portfolio demonstration using simulated data; it is not certified
for aircraft-maintenance decisions.
