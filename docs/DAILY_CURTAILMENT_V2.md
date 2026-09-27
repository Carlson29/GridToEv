# Optional v2: daily curtailment planning model

This is a **different task** from v1. v1 predicts half-hour renewable *dispatch-down* 30 or 60 minutes ahead. This experimental model, issued at **00:00 UTC on day D**, estimates (1) whether any *curtailment* will occur during D and (2) total curtailment energy in MWh during D. Its MAE must not be compared numerically with v1's half-hour dispatch-down MAE. It does not replace v1's artifact or endpoints.

## Inputs and leakage boundary

- Labels: complete 48-row UTC days of EirGrid half-hourly `curtailment_mwh` from `data/processed/eirgrid_core_history_30min.csv.gz`. These values are **never prediction features**.
- Forecasts: archived Open-Meteo Previous Runs `gfs_global` hourly 100 m wind speed, shortwave radiation, and 2 m temperature for Galway, Cork, Dublin, and Belfast. `previous_day1` denotes the value forecast 24 hours before each valid hour. For a target day D, the last D forecast is therefore available at D−1 23:00 UTC, before the D 00:00 issue. This is a fixed-lead-time source convention, not independently signed publication timestamps.
- Features: each region's daily mean, maximum and 75th-percentile wind speed; summed forecast solar radiation; mean forecast temperature; cross-region wind spread; and date seasonality/weekend. No future EirGrid measurement, realised weather, or lagged curtailment label is used.
- Validation rejects a day if any region lacks one of its 24 forecast hours, any forecast field is null, the forecast lead-time cutoff is breached, or any of the 48 EirGrid label intervals is missing or invalid.

The selected model has a gradient-boosted event classifier and an Extra Trees regressor trained on positive-curtailment days. Estimated daily MWh is event probability × estimated MWh conditional on an event. A direct-median amount model is also evaluated on 2025 validation and discarded unless it wins. The model always returns a probability and a nonnegative MWh estimate; a tiny nonzero MWh estimate does **not** mean an event is guaranteed.

## Reproduce

From the repository root, with the package installed:

```powershell
python scripts/build_daily_curtailment_data.py
python scripts/train_daily_curtailment_model.py
python -m unittest discover -s tests -v
```

The first command caches public source JSON in git-ignored `data/raw/open_meteo_previous_runs/gfs_global/` and writes the clean, committed `data/processed/daily_curtailment_forecast_v2.csv.gz`. The second writes `models/v2/daily_curtailment_bundle.joblib` and `benchmarks/daily_curtailment_v2/evaluation.json`. Data are split chronologically: 2024-04–12 training, all 2025 model selection, and 2026-01–08 untouched test. The saved fitted model has seen training plus validation data, **not test labels**. See the JSON report for exact row counts, hashes, methods and subgroup errors.

## Opt-in API

Set `GRIDTOEV_DAILY_MODEL_PATH=models/v2/daily_curtailment_bundle.joblib` in the API environment and start FastAPI as usual. The loader verifies the artifact SHA-256 against `benchmarks/daily_curtailment_v2/evaluation.json` **before** deserializing it, and verifies the bundled model dataset checksum before V2 serves historical windows. `GRIDTOEV_DAILY_REPORT_PATH` and `GRIDTOEV_DAILY_DATASET_PATH` can select those files if they live elsewhere. Without the model setting, all v1 endpoints continue to work and the daily endpoints return 503.

The Docker image leaves `GRIDTOEV_DAILY_MODEL_PATH` unset, while the live Render Blueprint sets it to `/app/models/v2/daily_curtailment_bundle.joblib`. Docker Compose still passes the setting through only when explicitly supplied by the operator. `GET /health` reports `daily_model_available` so operators can confirm that the optional model loaded without changing v1 readiness. To turn v2 off on Render, remove or clear the variable in the Blueprint and sync/redeploy; the v1 artifact and routes do not change.

If the optional bundle or report fails validation, v1 still starts and serves normally; only the v2 routes return 503.

The new endpoints are:

- `GET /dataset/daily-curtailment/coverage` — historical train/validation/test date ranges and complete-day counts, fitted-through date, dataset-window dates, and the separate currently requestable single-day archive range. This does **not** claim every later date has a complete archived weather forecast.
- `GET /model-info/daily-curtailment` — version, fitted classifier/regressor names and roles, feature list, candidate validation scores, full-day test metrics and baselines. Dates used to fit the model are not independent tests.
- `GET /model-info/daily-curtailment/raw-input-schema` — plain-language guide to the original hourly forecast fields and four regions.
- `POST /predict/curtailment/day` with `{"target_date_utc":"YYYY-MM-DD"}` — probability of at least some curtailment and predicted total MWh for that UTC day.
- `POST /predict/curtailment/from-raw` — supply 24 hourly forecast rows for each of the four regions (96 total); the API builds daily features and predicts that target UTC day. Forecast publication times must be known by issue, but caller-supplied provenance is not independently verified.
- `POST /predict/curtailment/window` with `{"start_date_utc":"2026-04-28","days":7}` — one historical replay prediction per complete V2 model-dataset day, April 28–May 4 in this example. Future or missing dataset dates are rejected.
- `GET /actuals/daily-curtailment?target_date_utc=YYYY-MM-DD` — the observed complete-day curtailment after it appears in the EirGrid archive; see `docs/ACTUALS_API.md`.

`GET /actuals/coverage` describes a **different** EirGrid actual-value snapshot. The V2 coverage
endpoint describes complete historical model-ready days (2024-04-01 through 2026-08-30, 882 days),
not when later observed outcomes become available.
The `/predict/curtailment/from-raw` Swagger default uses archived day-ahead
forecasts for **2026-08-31**, the first day after that model dataset. It sends
no observed curtailment and does not append a row to the dataset. It is a
one-step-out-of-dataset demonstration, not today's live weather forecast.

For example:

```powershell
$body = @{ target_date_utc = (Get-Date).ToUniversalTime().ToString('yyyy-MM-dd') } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/predict/curtailment/day -Method Post -ContentType application/json -Body $body
```

The target date must be between **2024-04-01 and the current UTC date**: earlier fixed-GFS archive coverage is incomplete, while tomorrow's full set of 24-hour-lead forecast values is not yet available at today's issue time. Live requests fetch four archived forecast snapshots from Open-Meteo; if the provider is unavailable, the API returns 503 rather than substituting today's observed weather. Requests need network access, and deployment must comply with the data provider's applicable usage terms. The Render Blueprint enables this experimental option on the live service; its forecasts should not be treated as dispatch instructions.

For `/predict/curtailment/window`, all 1–7 consecutive UTC dates must be in the bundled V2 model dataset (currently 2024-04-01 through 2026-08-30); inspect `window_date_min_utc` and `window_date_max_utc` in the coverage response. The route uses only that dataset's archived forecast/calendar feature columns, **not** its curtailment labels, and makes no weather-network call. This is a retrospective sequence of daily predictions, not a forecast of the next seven days from today. Dates through 2025-12-31 were used to fit the saved model and are in-sample; use the held-out report, not replay errors on those dates, to judge accuracy. The two raw-input Swagger demos remain the only examples that target the first unlabelled observation after their respective model datasets.

## Interpretation and release cautions

The daily model can help plan flexible EV charging or operator attention for the current UTC day. It is not a 48-point half-hour trajectory and not a precise dispatch instruction. Its estimate can be wrong by thousands of MWh on an individual high-curtailment day. An initial holdout comparison is against always-zero and monthly-median **daily** forecasts, not v1. Review the report's quarter-by-quarter errors before exposing this as a production option.

Open-Meteo documents the 24-hour-lead meaning of `previous_day1`, but does not provide immutable per-hour publication receipts in this API. Before production promotion, persist daily forecast payloads at issue time, audit that the retrospective values match those captured values, and monitor coverage, calibration, MAE and source-model changes. Keep the endpoint explicitly experimental until that check and a longer live shadow test pass.

Sources: [Open-Meteo Previous Runs documentation](https://open-meteo.com/en/docs/previous-runs-api), [EirGrid system and renewable data reports](https://www.eirgrid.ie/grid/system-and-renewable-data-reports).
