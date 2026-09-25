# Benchmark artifacts

Issue #1 establishes the comparison contract that every later model experiment must use. It exists to
stop a candidate from appearing better because it used different dates, saw future information, or was
measured with a different metric.

## Reproduce the frozen baseline

From an installed project environment, run:

```powershell
python scripts/run_benchmark.py
```

The command does not overwrite the production model. It writes:

- `v1.1.0/benchmark_report.json`: full partition, fold, horizon, event-regime, classification,
  regression, and uncertainty results;
- `v1.1.0/experiment_registry.csv`: one compact baseline record designed to receive future candidate
  rows; and
- no fitted model artifact, because v1.1.0 remains the rollback bundle under `models/`.

## Frozen evaluation contract

`config/benchmark_contract.v1.json` is the source of truth for:

- the 70% training, 15% validation, and sealed 15% final-test split;
- the 40%-60%, 60%-80%, 80%-100% expanding windows inside training, followed by the validation fold;
- release thresholds and baseline metrics;
- the dataset, model, metrics, and metadata SHA-256 hashes; and
- the exact final-test dates and row count.

The operating-policy selector accepts only training and validation frames. The final test is supplied to
a separate scoring function after the probability threshold, horizon-specific trend/blend settings, and
uncertainty adjustment have been frozen.

## Leakage and integrity gates

The runner fails before training when it finds:

- rows not sorted by issue time and forecast horizon;
- duplicate issue-time/horizon keys;
- targets that are not strictly in the future or do not match their stated horizon; or
- a `published_at_utc` or `source_vintage_utc` feature timestamp later than the issue time.

These checks intentionally fail closed. New timestamped source families must preserve publication or
vintage columns so they receive the same test.

## Frozen v1.1.0 result

The reproducible final-test result is:

- dispatch-down MAE: 11.825026 MWh;
- RMSE: 20.590388 MWh;
- WAPE: 17.217796%;
- event average precision: 0.999795;
- P50 pinball loss: 6.997639; and
- P10-P90 empirical coverage: 74.477958%.

Future candidate rows should retain the same `contract_id`. A new contract version is required if the
dataset, folds, final-test period, metric definitions, or leakage rules intentionally change.

## Horizon-specific candidate benchmark

Issue #8 writes its standard-model and stable-ensemble comparison under `horizon_models/`. The best
development candidate improves rolling MAE by 1.22%, but it fails the 15% aggregate gate and the fold-
stability rule. The final test is therefore not accessed and v1.1.0 remains the active model. See
`docs/HORIZON_MODEL_BENCHMARK.md` for the full decision and reproduction command.

## SEMO market-signal ablation

Issue #5 records its named feature-family decision under `semo_market_signals/`. The retained SEMO
high-frequency archive does not overlap the frozen January 2026 labelled benchmark, so the current
record has null metrics, does not access the final test, and excludes the family from production. This
explicit rejection avoids claiming an improvement from non-overlapping data and can be replaced by a
measured candidate after enough overlapping publication vintages and labels have accumulated.
