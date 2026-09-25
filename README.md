# GridToEV dispatch-down forecasting

This repository prepares energy data, trains the GridToEV forecasting models, and serves predictions
through FastAPI. It forecasts renewable dispatch-down in Ireland 30 or 60 minutes ahead.

The notebooks are:

- `notebooks/01_build_model_ready_dataset.ipynb`
- `notebooks/02_train_and_export_models.ipynb`
- `notebooks/03_extend_history_and_forecast_vintages.ipynb`
- `notebooks/05_build_semo_market_signals.ipynb`
- `notebooks/08_benchmark_horizon_models.ipynb`

It writes one combined modelling table:

- `data/processed/gridtoev_model_ready.csv`

The current build contains 2,867 rows and 133 columns from 2 January through 31 January 2026. Each
issue time appears once for the 30-minute horizon and once for the 60-minute horizon. The CSV has no
missing cells and no duplicate issue-time/horizon keys.

## Quick start

Create a Python environment and install the package, notebook, and test dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run the data notebook first, then run the training notebook. The first downloads any missing public
source files and rebuilds the combined table. The second trains all seven models and writes the trusted
bundle to `models/gridtoev_model_bundle.joblib`.

The same training can be run without Jupyter:

```powershell
python scripts/train_models.py
```

Reproduce and audit the frozen v1.1.0 benchmark without replacing the production model:

```powershell
python scripts/run_benchmark.py
```

That command validates dataset order and publication times, selects operating settings without final-
test access, scores the sealed test only after selection, verifies the committed hashes and metrics,
and writes the granular report and experiment registry under `benchmarks/v1.1.0/`.

Compare horizon-specific linear, histogram-gradient-boosting, Extra Trees, two-stage, and stable
out-of-fold ensemble candidates:

```powershell
python scripts/run_horizon_benchmark.py
```

The current best candidate improves rolling MAE by only 1.22% and fails the fold-stability gate, so
v1.1.0 remains active and the final test stays sealed. See `docs/HORIZON_MODEL_BENCHMARK.md`.

Build the multi-year EirGrid history and leakage-safe forecast-vintage tables:

```powershell
python scripts/build_extended_data.py --forecast-days 3
```

The command is resumable: existing raw files are checksum-verified and not downloaded again. It
builds 2021-present half-hourly system/dispatch history, collects retained SEMO wind, demand, and
interconnector forecast publications, snapshots the EirGrid solar forecast, and performs backward
as-of joins for 30- and 60-minute horizons. See `docs/EXTENDED_DATA_PIPELINE.md` for the schema,
quality gates, current measured coverage, and source limitations.

Build the SEMO market and operational signal family:

```powershell
python scripts/build_semo_market_data.py --days 2 --workers 12
```

This produces auditable, publication-time-safe schedule, PN, forecast-imbalance, NTC, lagged
imbalance-price, reserve, and SSII/SIFF features. See `docs/SEMO_MARKET_SIGNALS.md` for the full data
contract, outputs, leakage controls, and current ablation decision.

Start the prediction API:

```powershell
uvicorn gridtoev.api:app --app-dir src --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/docs` for interactive API documentation. A frontend example is available
in `examples/frontend-prediction.js`.

To run the output checks:

```powershell
python -m unittest discover -s tests -v
```

## Output files

- `gridtoev_model_ready.csv`: the single combined modelling dataset.
- `gridtoev_data_dictionary.csv`: column role, unit, source, and data type.
- `gridtoev_quality_report.json`: row count, coverage, event rates, and quality checks.
- `source_manifest.csv`: source URLs, local paths, file sizes, and SHA-256 hashes.
- `models/gridtoev_model_bundle.joblib`: fitted preprocessing and seven prediction models.
- `models/model_metadata.json`: feature contract, versions, split dates, and model settings.
- `models/training_metrics.json`: chronological validation and test results.
- `config/benchmark_contract.v1.json`: versioned splits, folds, release gates, hashes, and expected
  v1.1.0 metrics.
- `benchmarks/v1.1.0/benchmark_report.json`: aggregate, per-horizon, per-fold, and event-regime results.
- `benchmarks/v1.1.0/experiment_registry.csv`: machine-readable baseline row for future comparisons.
- `benchmarks/horizon_models/benchmark_report.json`: Issue #8 candidate, horizon, fold, regime and
  release-gate report.
- `benchmarks/horizon_models/experiment_registry.csv`: machine-readable candidate and ablation runs.
- `benchmarks/horizon_models/development_oof_predictions.csv.gz`: auditable development predictions
  used to choose ensemble weights.
- `eirgrid_core_history_30min.csv.gz`: 2021-present half-hourly EirGrid history with availability
  flags; gzip is read directly by pandas.
- `forecast_vintages.csv.gz`: long-form target/publication/retrieval-time forecast revisions.
- `forecast_features_asof_30_60.csv`: leakage-safe 30/60-minute forecast feature matrix.
- `forecast_model_features_30_60.csv`: causal forecast-error, renewable-state and grid-headroom table.
- `forecast_model_feature_dictionary.csv`: formula, source, unit and availability rule for every column.
- `forecast_model_feature_quality_report.json`: Issue #4 leakage, key, coverage and missingness checks.
- `extended_source_manifest.csv`: provider, report, URL, retrieval time, checksum, row count, schema,
  and per-file coverage for the extended sources.
- `semo_market_signals.csv.gz`: normalized long-form SEMO market and operational signals.
- `semo_market_features_asof_30_60.csv`: leakage-safe SEMO features for each issue time and horizon.
- `semo_market_feature_dictionary.csv`: complete SEMO feature definitions and availability rules.
- `semo_market_source_manifest.csv`: source URL, retrieval timestamp, checksum, schema, and coverage.
- `semo_market_quality_report.json`: per-report coverage, staleness, missingness, and leakage checks.
- `benchmarks/semo_market_signals/ablation_report.json`: named feature-family evaluation decision.

## Prediction API

- `GET /health`: readiness and model version.
- `GET /model-info`: training metadata and available dataset time range.
- `GET /dataset/available-times`: timestamps for a frontend selector.
- `GET /predict/latest`: both forecast horizons for the latest available issue time.
- `POST /predict/from-dataset`: one selected historical time and horizon.
- `POST /predict/features`: a complete live feature snapshot.

See `docs/HOW_IT_WORKS.md` for the end-to-end explanation and request flow.
See `docs/FORECAST_FEATURE_ENGINEERING.md` for the Issue #4 feature formulas and leakage controls.

## 48-hour performance sprint

The performance sprint is coordinated in
[#11](https://github.com/Carlson29/GridToEv/issues/11). The complete schedule, public-data source
matrix, leakage controls, release targets and issue acceptance criteria are in
[`docs/48_HOUR_MODEL_PERFORMANCE_PLAN.md`](docs/48_HOUR_MODEL_PERFORMANCE_PLAN.md).

## Modelling structure

The natural key is `issue_timestamp_utc` plus `forecast_horizon_minutes`. The target time is stored
separately in `target_timestamp_utc`.

The main targets are:

- `dispatch_down_event`: whether future dispatch-down is greater than zero.
- `dispatch_down_mwh`: future dispatched-down renewable energy.
- `curtailment_mwh`: future system-wide curtailment.
- `constraint_mwh`: future network constraint energy.
- `recoverable_surplus_upper_bound_mwh`: dispatch-down energy available in principle.
- `recoverable_surplus_100mw_flex_mwh`: recoverable energy under a clearly labelled 100 MW flexible
  load scenario for one half-hour.

Feature groups include generation mix, load, price, wind and solar availability, renewable share,
net load, SNSP headroom, interconnector flows, oversupply, calendar cycles, 30-minute to 24-hour
lags, ramps, rolling statistics, the latest completed dispatch-down interval, and older dispatch-down
history.

## Leakage controls

- Future labels are shifted by one or two half-hour steps before they are attached to feature rows.
- The latest completed dispatch-down interval is an issue-time feature; every label is attached to a
  strictly later target timestamp.
- Older dispatch-down observations remain as lagged features for trend estimation.
- Rolling statistics end one complete interval before the issue time.
- The quality gate verifies timestamp ordering and target accounting.
- The benchmark gate rejects shuffled rows, misaligned targets, and feature publication/vintage times
  later than the model issue time.
- Model evaluation uses the versioned chronological split and rolling-origin folds, never a random
  train/test split.

## Scope and limitations

The organiser's Irish price sample is hourly and covers January 2026, so the fully populated
multi-source table uses that shared January window. Hourly prices are carried forward once to the
half-hour grid and flagged. Short generation/load gaps are interpolated only across at most one hour
and are also flagged. The notebook does not invent unavailable load or wind forecasts.

Raw downloads are excluded from version control to avoid duplicating upstream data. Large processed
tables are stored as deterministic gzip CSV files. The notebook, catalog, and source manifest make the
build reproducible. Review upstream terms before redistributing raw files.
