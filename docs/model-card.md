# EngineGuard Model Card

## Model

Frozen XGBoost Regressor trained on all 80 FD001 development engines.

- Feature schema: `v1`
- Random seed: `42`
- Final test RMSE: `14.2892`
- Final test MAE: `10.4291`

## Intended use

Portfolio demonstration of RUL prediction on simulated NASA C-MAPSS FD001 data. It is not certified for aircraft-maintenance decisions.

## Limitations

FD001 contains one operating condition and one fault mode. SHAP explains model behavior for these inputs; it does not establish physical sensor causality.
