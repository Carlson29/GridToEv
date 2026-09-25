# Horizon-specific models and stable ensembles

Issue #8 compares standard dispatch-down model families under the frozen
`dispatch-down-benchmark-v1` contract. It does not overwrite the production bundle.

## What was compared

Every learned candidate fits a separate 30-minute and 60-minute model on the same four expanding
time folds used by v1.1.0. The comparison includes:

- frozen v1.1.0, latest-observation and stale-persistence baselines;
- regularised Ridge residual models;
- histogram gradient-boosting residual models;
- Extra Trees residual models;
- a two-stage event-probability times positive-quantity model; and
- pairwise, horizon-specific ensembles selected only from development out-of-fold predictions.

Optional LightGBM, XGBoost and CatBoost dependencies were not added. The standard candidates ran
reliably but did not approach the 15% development improvement gate, so adding another dependency had
no demonstrated incremental value within this experiment.

## Leakage and selection rules

The runner calls the existing benchmark validator before fitting. It uses the exact chronological
split, natural keys and rolling folds from `config/benchmark_contract.v1.json`.

Candidate settings and ensemble weights use only the rolling development predictions. A candidate
must satisfy all three conditions before the sealed final test can be scored:

1. aggregate rolling MAE improves by at least 15%;
2. neither forecast horizon regresses; and
3. at least three of four folds improve, with no fold degrading by more than 2%.

This last rule rejects a model that wins only because one period was unusually favourable.

## Result and production decision

The best candidate was an out-of-fold ensemble of v1.1.0 and the Extra Trees residual model. Its
weights were selected independently by horizon:

- 30 minutes: 85% v1.1.0 and 15% Extra Trees;
- 60 minutes: 65% v1.1.0 and 35% Extra Trees.

Development MAE changed from **9.8717 MWh** to **9.7515 MWh**, a **1.22% improvement**. Both horizons
improved slightly, but only two of four folds improved. The two regressions were approximately 4.49%
and 2.65%, which exceed the 2% fold tolerance.

The candidate therefore failed the development gate. The final test was not accessed, no production
artifact was written, and **v1.1.0 remains active**. This is a useful negative result: a small aggregate
gain is not stable enough to justify changing the prediction service.

## Feature-family ablations

The strongest individual learner, histogram gradient boosting, was rerun with:

- all features;
- dispatch-down persistence/history only;
- renewable and grid-state signals;
- demand, market and interconnector signals;
- temporal, lag, ramp and rolling signals; and
- all features except dispatch-down history.

None beat v1.1.0. The report also includes no-event, positive-event, high-dispatch, ramp-up and
ramp-down error slices, fold dispersion, training time and inference time for every candidate.

## Reproduce the benchmark

From an installed project environment:

```powershell
python scripts/run_horizon_benchmark.py
```

Outputs:

- `benchmarks/horizon_models/benchmark_report.json`
- `benchmarks/horizon_models/experiment_registry.csv`
- `benchmarks/horizon_models/development_oof_predictions.csv.gz`
- `notebooks/08_benchmark_horizon_models.ipynb`
