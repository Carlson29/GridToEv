# Trained model bundle

`gridtoev_model_bundle.joblib` is the trusted model artifact loaded by the FastAPI service. Rebuild it
with either `notebooks/02_train_and_export_models.ipynb` or `python scripts/train_models.py`.

The bundle contains:

- an event classifier for the probability of dispatch-down;
- an MWh change regressor blended with the latest observed dispatch-down value;
- separate curtailment and network-constraint regressors; and
- P10, P50, and P90 quantile regressors with a validation-calibrated interval adjustment.

`model_metadata.json` records the exact 117-column feature order, dataset hash, chronological split,
model settings, threshold, blend weight, and package versions. `training_metrics.json` contains the
held-out evaluation results.

## Current evaluation summary

The final chronological test period is 27-31 January 2026.

- Event classifier: average precision 0.9995, F1 0.9966, Brier score 0.0092.
- Hybrid dispatch-down regression: MAE 18.85 MWh and WAPE 27.45%.
- Persistence baseline: MAE 16.82 MWh and WAPE 24.49%.
- Calibrated P10-P90 interval coverage: 71.46%.

The event classifier adds clear value over the persistence event baseline. The MWh machine-learning
component did not beat pure persistence on this short final test period, so the API keeps persistence
inside the hybrid and reports the comparison openly. The test period has no positive curtailment MWh,
so curtailment WAPE is intentionally reported as `null`; collect a longer, seasonally diverse history
before treating that component as production-ready.

Only load this artifact from the trusted repository or a verified build. Joblib uses pickle-compatible
serialization and must not be used to load untrusted files.
