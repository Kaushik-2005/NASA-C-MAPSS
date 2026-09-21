# FD001 Data Dictionary

## Dataset scope

| Property | Value |
|---|---|
| Dataset | NASA C-MAPSS FD001 |
| Training engines | 100 |
| Test engines | 100 |
| Operating conditions | 1 |
| Fault modes | 1 |
| Training rows | 20,631 |
| Test rows | 13,096 |

## Column groups

| Group | Columns | Meaning |
|---|---|---|
| Identity | `unit_id` | Engine identifier |
| Time | `cycle` | Operating cycle number |
| Operating settings | `op_setting_1`–`op_setting_3` | Simulated operating-condition values |
| Sensors | `sensor_1`–`sensor_21` | Numeric engine measurements |

The raw trajectory rows contain 26 numeric fields in the fixed order defined
by `src.data.schema.EXPECTED_COLUMNS`. Trailing whitespace is removed by the
whitespace parser without modifying the immutable raw files.

## Target and derived labels

For training engines:

```text
raw_rul = maximum_training_cycle_for_engine - current_cycle
target_rul = min(raw_rul, 125)
failure_within_30 = 1 if raw_rul <= 30 else 0
```

`target_rul` is the regression training target. `failure_within_30` is a
learning-only classification target and is not served by the final API.

## Feature-selection exclusions

Variance analysis was fit using development engines only (`unit_id % 5 != 0`)
with a threshold of `1e-6`. The recorded excluded list is maintained in
`configs/excluded_features.json`:

- `sensor_1`
- `sensor_5`
- `sensor_10`
- `sensor_16`
- `sensor_18`
- `sensor_19`

This list is an engineering feature-selection decision, not evidence that the
sensors are physically irrelevant.

## Risk mapping

| Predicted RUL | Risk level |
|---|---|
| 0–30 cycles | `critical` |
| 31–60 cycles | `warning` |
| 61–125 cycles | `healthy` |

Risk is deterministic application logic derived from clipped predicted RUL;
it is not a separately trained classifier.

## Limitations

- FD001 contains one operating condition and one fault mode.
- The data is simulated and does not represent certified aircraft-maintenance data.
- Sensor numbers do not establish physical meaning or causal importance.
- Correlation and SHAP explanations describe model behavior, not physical causality.
- Results cannot automatically generalize to FD002, FD003, FD004, or real engines.
- The official test labels remain isolated until the final evaluation module.
