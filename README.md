# GridToEV dispatch-down forecasting

This repository prepares energy data, trains the GridToEV forecasting models, and serves predictions
through FastAPI. It forecasts renewable dispatch-down in Ireland 30 or 60 minutes ahead.

The two executed notebooks are:

- `notebooks/01_build_model_ready_dataset.ipynb`
- `notebooks/02_train_and_export_models.ipynb`

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

## Prediction API

- `GET /health`: readiness and model version.
- `GET /model-info`: training metadata and available dataset time range.
- `GET /dataset/available-times`: timestamps for a frontend selector.
- `GET /predict/latest`: both forecast horizons for the latest available issue time.
- `POST /predict/from-dataset`: one selected historical time and horizon.
- `POST /predict/features`: a complete live feature snapshot.

See `docs/HOW_IT_WORKS.md` for the end-to-end explanation and request flow.

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
- Model evaluation should use chronological splits, never a random train/test split.

## Scope and limitations

The organiser's Irish price sample is hourly and covers January 2026, so the fully populated
multi-source table uses that shared January window. Hourly prices are carried forward once to the
half-hour grid and flagged. Short generation/load gaps are interpolated only across at most one hour
and are also flagged. The notebook does not invent unavailable load or wind forecasts.

Raw downloads are excluded from version control to avoid duplicating upstream data. The notebook and
source manifest make the build reproducible. Review upstream terms before redistributing raw files.
