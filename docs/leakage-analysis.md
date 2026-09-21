# Module 5 - Leakage analysis

## Target construction

For each training engine and cycle:

```text
raw_rul = maximum_training_cycle_for_engine - current_cycle
target_rul = min(raw_rul, 125)
failure_within_30 = 1 if raw_rul <= 30 else 0
```

The labels are created from training trajectories only. They are never passed
to the feature builder as model inputs.

## Engine-level partitions

Rows are never split randomly. Engine IDs are assigned deterministically:

| Partition | Rule | Engines |
|---|---|---:|
| Validation | `unit_id % 5 == 0` | 20 |
| Development | `unit_id % 5 != 0` | 80 |
| Initial training | `unit_id <= 75 and unit_id % 5 != 0` | 60 |
| Incremental training | `unit_id > 75 and unit_id % 5 != 0` | 20 |

Development and validation are disjoint. Initial and incremental training are
disjoint subsets of development, and their union equals development.

## Sample policy

- Training samples include eligible development rows from cycle 20 onward.
- Validation samples use cycles at 60%, 70%, 80%, and 90% of each validation
  engine's lifetime, rounded down and bounded at cycle 20.
- Duplicate validation cutoffs are counted and removed per engine.
- The final test set contributes one final available row per engine.

Measured FD001 artifacts:

| Sample set | Count |
|---|---:|
| Training samples | 15,136 |
| Validation samples | 80 |
| Requested validation cutoffs | 80 |
| Deduplicated validation cutoffs | 0 |
| Final test samples | 100 |

## Forbidden feature inputs

The following values are forbidden as model features:

- `raw_rul`
- `target_rul`
- `failure_within_30`
- Maximum lifecycle or maximum training cycle
- Future rows from the same engine
- Official test RUL labels

The implementation contains automated assertions for forbidden label/lifecycle
columns and for sample membership in the declared engine manifest.

## Test-set isolation

`RUL_FD001.txt` is loaded only for structural validation at this stage. It has
not influenced feature selection, preprocessing, model choice, hyperparameters,
thresholds, or acceptance criteria. Official test evaluation remains reserved
for Module 9.

## Evidence

- `data/manifests/engine_manifests.json`
- `data/manifests/sample_manifest.json`
- `src/data/leakage.py`
- `tests/unit/test_leakage.py`
- `tests/unit/test_samples.py`
