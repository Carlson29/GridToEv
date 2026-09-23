# Trained model bundle

`gridtoev_model_bundle.joblib` is the trusted model artifact loaded by the FastAPI service. Rebuild it
with either `notebooks/02_train_and_export_models.ipynb` or `python scripts/train_models.py`.

The bundle contains:

- an event classifier for the probability of dispatch-down;
- a horizon-specific damped trend with a guarded MWh residual regressor;
- separate curtailment and network-constraint regressors; and
- P10, P50, and P90 quantile regressors with a validation-calibrated interval adjustment.

`model_metadata.json` records the exact 119-column feature order, dataset hash, chronological split,
model settings, threshold, horizon-specific trend and blend settings, and package versions.
`training_metrics.json` contains the held-out evaluation results.

## Current evaluation summary

The final chronological test period is 27-31 January 2026.

- Event classifier: average precision 0.9998, F1 0.9899, Brier score 0.0097.
- Guarded dispatch-down forecast: MAE 11.83 MWh and WAPE 17.22%.
- Older one-interval-stale persistence baseline: MAE 16.82 MWh and WAPE 24.49%.
- P50 pinball loss: 7.00, down from 34.34 before residual centring.
- Calibrated P10-P90 interval coverage: 74.48%.

The event classifier adds value over the event persistence baseline. The central MWh forecast uses the
latest completed observation plus a rolling-backtest-selected damped trend, reducing MAE by 29.7%
against the older stale baseline. The boosted-tree residual correction receives zero weight for both
horizons in version 1.1.0 because it did not improve every backtest fold. The test period has no
positive curtailment MWh, so curtailment WAPE is intentionally `null`; collect a longer, seasonally
diverse history before treating that component as production-ready.

Only load this artifact from the trusted repository or a verified build. Joblib uses pickle-compatible
serialization and must not be used to load untrusted files.
