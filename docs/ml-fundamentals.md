# Module 2 — ML foundations

## Concepts

EngineGuard's RUL task is regression: predict a continuous cycle count. The
derived `failure_within_30` task is binary classification. Features must be
computed from the current and earlier history only; a random row split would
place observations from the same engine in both partitions and make validation
too optimistic.

## Formulas

Linear regression minimizes mean squared error:

```text
MSE = (1/n) * sum((y - Xw)^2)
w <- w - learning_rate * gradient(MSE)
```

Ridge adds `lambda * sum(w_j^2)` for feature weights, excluding the intercept.
Logistic regression maps a linear score through `sigmoid(z) = 1/(1+exp(-z))`
and minimizes binary cross-entropy (log loss).

## EngineGuard application

The NumPy implementations in `src/foundations/` make the optimization steps
visible before using scikit-learn. A continuous RUL target needs regression;
the 30-cycle label is useful for learning classification metrics and threshold
trade-offs but is not the deployed risk model.

## Trade-offs and failure modes

Gradient descent is transparent but sensitive to feature scale and learning
rate. Ridge reduces coefficient variance under correlated sensors but can
underfit nonlinear degradation. Logistic thresholds trade recall for precision;
missing a near-term failure is operationally more costly than an unnecessary
warning, so recall should be examined explicitly.

## Interview questions

1. Why not randomly split rows? Because the same engine's temporal signature
   can leak across partitions; split by engine identity.
2. Why standardize for Ridge? A penalty acts on coefficient magnitude, so
   unscaled features receive unequal effective regularization.
3. Why is accuracy insufficient for `failure_within_30`? Near-term failures
   may be the minority class, allowing a majority-class model to look accurate
   while missing failures.

## FD001-backed verification

The reproducible follow-up is implemented in
`scripts/verify_foundations_fd001.py` and called by the final cell in
`notebooks/01_ml_foundations.ipynb`. It uses only the 80 development engines
and 16,656 development rows, with `cycle` and `sensor_2` as the small feature
subset. The measured comparison is stored in
`reports/foundations-fd001.json`; the custom regression and classification
implementations both decrease their training loss and show the same behavior
as their scikit-learn counterparts.
