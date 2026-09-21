# Module 7 Baseline Results

These results use only the FD001 development engines for training and the fixed 20-engine validation partition for evaluation. Official test labels were not used.

- Development samples: 15136
- Validation samples: 80
- Standardized selected features: 133

## Regression

| Model | RMSE | MAE | R-squared | Train ms | Inference ms | Size bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| median | 50.2057 | 44.7375 | -3.2399 | 2.757 | 0.022 | 90 |
| ridge | 20.4961 | 17.0469 | 0.2934 | 231.294 | 1.153 | 1713 |

## Logistic classification

The classifier predicts `failure_within_30`. PR-AUC is reported as average precision.

| Threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC | Confusion matrix |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.50 | 0.9500 | 0.9048 | 0.9268 | 0.9871 | 0.9721 | `[[58, 1], [2, 19]]` |
| 0.30 | 0.9048 | 0.9048 | 0.9048 | 0.9871 | 0.9721 | `[[57, 2], [2, 19]]` |

- Logistic training time: 637.266 ms
- Logistic inference time: 1.674 ms
- Logistic serialized size: 2031 bytes

## Limitations

- These are fixed validation results, not official FD001 test results.
- MLflow status: logged.
- Timing is a local single-run measurement and is not a production latency claim.
