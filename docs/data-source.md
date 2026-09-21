# FD001 data source and checksums

The official NASA Open Data record identifies the C-MAPSS archive and the
FD001 composition: 100 training engines, 100 test engines, one operating
condition, and one fault mode.

The official archive download was attempted from:

```text
https://data.nasa.gov/docs/legacy/CMAPSSData.zip
```

Because the local transfer repeatedly truncated the archive, the three
required FD001 files were downloaded individually from this public mirror:

```text
https://github.com/bishals098/NASA-CMAPSS-dataset
```

Files were stored unchanged under `data/raw/`. Validation observed:

| File | Lines | Engines/values | SHA-256 |
| --- | ---: | ---: | --- |
| `train_FD001.txt` | 20,631 | 100 engines | `963B5E22825B34D8B21C69E1AEB4AF3E647050EB672EE8834BA4B5D91D2DE0F8` |
| `test_FD001.txt` | 13,096 | 100 engines | `3CDA7109CE17BAFB5443F2AC926CFCF88154B941B8C4CF95EB55D1DDD6F52851` |
| `RUL_FD001.txt` | 100 | 100 values | `A19C8EC94931949D0485BDC35118206E9C81C4547B422EFB9CF86F4CEDDBCECA` |

Trajectory rows contain trailing spaces. After removing empty fields caused by
whitespace parsing, each row has the required 26 fields. The raw files are not
modified by the project.
