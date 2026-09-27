# Share the GridToEV model and request predictions

## What colleagues receive

The `release/shareable-prediction-api` branch is the serving branch. It contains the frozen V1.1.0
30/60-minute dispatch-down model, its historical demonstration dataset, and an independently loaded,
experimental V2 daily-curtailment model. Prediction requests do not run notebooks or retrain either
model. V2 is enabled by the live Render Blueprint; a local Docker Compose deployment remains opt-in.

The easiest interface is the interactive page at `BASE_URL/docs`. The machine-readable endpoints
return JSON and can be called from a frontend, Python, PowerShell, curl, or another backend.

## Option A: one shared internet URL

This is the recommended hackathon setup because only one person operates the server.

1. Push or select the `release/shareable-prediction-api` branch in GitHub.
2. Sign in to Render and create a new **Blueprint** from `Carlson29/GridToEv`.
3. Select `release/shareable-prediction-api` as the Blueprint branch. Render reads `render.yaml` and
   builds the included `Dockerfile`.
4. For `GRIDTOEV_API_KEY`, enter a long random team key. Generate one locally with:

   ```powershell
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

5. Set `GRIDTOEV_CORS_ORIGINS` to the exact frontend origins, separated by commas, for example
   `https://gridtoev.example.com,http://localhost:5173`. If colleagues only use `/docs`, Python,
   PowerShell, or curl, this value does not affect them.
6. Deploy, wait until `/health` returns `{"status":"ready",...}`, then check
   `daily_model_available` if the daily V2 option is expected. Share only these two values
   through a private team channel:

   - the Render URL, such as `https://gridtoev-api.onrender.com`;
   - the API key.

Do not commit the key. A free service may sleep when idle, so its first request can be slower. Render
uses `/health` to reject a deployment that cannot load the model.
V1 readiness is independent of V2: `/health` can still say `ready` when
`daily_model_available` is false.

## Option B: run it on one teammate's computer

This works on the same Wi-Fi or VPN and needs no cloud account.

1. Clone the serving branch and enter the repository:

   ```powershell
   git clone --branch release/shareable-prediction-api https://github.com/Carlson29/GridToEv.git
   Set-Location GridToEv
   ```

2. Start the container. An API key is strongly recommended when other people can reach the machine:

   ```powershell
   $env:GRIDTOEV_API_KEY = "paste-a-long-team-key-here"
   docker compose up --build -d
   ```

3. Find the host computer's IPv4 address with `ipconfig`. A colleague on the same network uses
   `http://HOST_IP:8000/docs`. Windows may ask the host to allow inbound port 8000; permit it only on
   the intended private network.
4. Stop the service later with `docker compose down`.

Without Docker, create a Python 3.12 environment, install `requirements.txt`, and run:

```powershell
$env:GRIDTOEV_API_KEY = "paste-a-long-team-key-here"
python -m gridtoev.server
```

## Make a prediction in the interactive page

1. Open `BASE_URL/docs`.
2. If the server uses a key, select **Authorize**, paste the key into `X-API-Key`, and confirm.
3. Expand `GET /predict/latest`, select **Try it out**, optionally change
   `flexible_load_capacity_mw`, and select **Execute**. This returns both the 30- and 60-minute
   forecasts for the newest row in the bundled demonstration dataset.
4. To choose an historical interval, call `GET /dataset/available-times`, copy one timestamp, then
   send it to `POST /predict/from-dataset` with a horizon of `30` or `60`.
5. For an array covering 2 or up to 24 hours of historical issue times, call
   `POST /predict/window/from-dataset`, supply a valid `start_timestamp_utc`, and use
   `duration_hours: 2` or `24`.

The bundled data is historical, so `latest` means the newest timestamp in that file, not live grid
conditions. True live prediction requires a separate collector to create the full feature snapshot
accepted by `POST /predict/features`.

Before displaying a date picker, call `GET /dataset/info`. It returns the minimum and maximum date,
the exact `YYYY-MM-DDTHH:MM:SSZ` format, the valid `:00`/`:30` minute alignment, and supported
horizons. If a timestamp is unavailable, the API returns the range and nearest timestamps rather than
only saying “not found.”

### Two-hour or one-day historical window

```json
{
  "start_timestamp_utc": "2026-01-15T12:00:00Z",
  "duration_hours": 2,
  "forecast_horizons_minutes": [30, 60],
  "flexible_load_capacity_mw": 100
}
```

Send this body to `POST /predict/window/from-dataset`. A 2-hour request returns four half-hour issue
times, or eight prediction objects when both horizons are selected. A 24-hour request returns 48 issue
times, or 96 prediction objects with both horizons. The response also includes a separate summary
for each horizon so totals are not accidentally double-counted.

This is a rolling historical replay: every item is still a 30- or 60-minute forecast using the feature
row available at that item's issue time. It is useful for charts, demonstrations, and backtesting, but
it is not a single forecast made 2 or 24 hours ahead. Swagger now prefills a complete latest
historical 24-hour replay (`2026-01-30T22:30:00Z`, 24 hours, both horizons). Its 96 results
are past dataset replay predictions, not a forecast for the next 24 hours from today.

## Use experimental daily V2

This is **not** another V1 horizon. V2 predicts whether any curtailment will occur and the total
curtailment MWh over one UTC day. Call `GET /dataset/daily-curtailment/coverage` to see historical
train/validation/test coverage and the separate requestable-date range. Call
`GET /model-info/daily-curtailment` to see its version and daily test metrics.

Send this body to `POST /predict/curtailment/day` with the same `X-API-Key`:

```json
{"target_date_utc":"2026-08-30"}
```

Use `YYYY-MM-DD` from 2024-04-01 through the current UTC date. V2 obtains its own archived weather
forecast fields, so no horizon or feature object is needed. A date may be within the allowed range
yet fail if the forecast source lacks a complete day. To check the observed outcome later, copy the
returned `target_date_utc` to `GET /actuals/daily-curtailment`; `GET /actuals/coverage` shows which
complete UTC days have observed labels. V2 is experimental and should not be used as a dispatch
instruction.

## Predict from original source values

Both `/predict/v1/from-raw` and `/predict/curtailment/from-raw` are **POST**
routes: opening their URLs in a browser sends GET and does not make a
prediction. In `/docs`, select **Authorize**, enter `X-API-Key`, expand the
POST route, click **Try it out**, edit the JSON body, then click **Execute**.
Swagger prefills both routes to predict the **first target after each frozen
model dataset**, using the latest applicable inputs. The target and its actual
outcome are **not** inserted into either dataset:

- V1's last dataset issue is `2026-01-31T22:30:00Z` and its last recorded
  target is `23:00`. The default uses that latest source row, 48 prior
  half-hours and a **60-minute** horizon to predict `23:30`, the next
  unrecorded half-hour. Its publication times are illustrative because the
  dataset did not retain per-source release receipts.
- V2's model dataset ends `2026-08-30`. The default predicts `2026-08-31`,
  using 96 archived Open-Meteo GFS day-ahead forecast values for that day.
  No observed curtailment value is supplied.

These are one-step-out-of-dataset demos, not forecasts for the current clock
time. Changing only the date while retaining old readings or weather would
give a misleading prediction; replace the inputs and availability times too.

Start with `GET /models/catalog`, then inspect `GET /model-info` (V1) or
`GET /model-info/daily-curtailment` (V2). These now show each fitted estimator's
name and job, the held-out scores that were actually recorded, baselines, test
periods and limitations. V1 has seven scikit-learn histogram-gradient-boosting
estimators, but its selected dispatch-down MWh serving blend currently gives
the ML point regressor weight **zero** at both horizons; the selected trend
calculation supplies the final amount. V2 has a boosted event classifier and
an Extra Trees positive-day amount regressor. There is no deployed XGBoost
model. V1 half-hour dispatch-down MAE and V2 whole-day curtailment MAE are
different targets and must not be compared numerically.

For a manual V1 prediction, inspect `GET /model-info/v1/raw-input-schema`,
then send `POST /predict/v1/from-raw` with:

- `issue_timestamp_utc` on a UTC `:00`/`:30` boundary and a 30- or 60-minute horizon;
- `current_observation`: the source-level ENTSO-E/EirGrid values described by the schema,
  the **already observed** dispatch-down for the completed issue half-hour, and the
  latest publication time across those values;
- `history`: exactly 48 consecutive preceding half-hour source rows, oldest first,
  containing wind generation, demand, price, SNSP, oversupply and **past** observed
  dispatch-down plus their timestamps/availability times.

The API calculates sums, headroom, calendar values, lags, ramps and rolling
statistics itself; do not send engineered columns or the future target. It
rejects gaps, missing source values, post-issue availability and a half-hour
price that differs from the preceding hourly value. The trained model's median
imputer is **not** a substitute for missing raw history: this endpoint requires
complete source input and sets the source-imputation flags to zero. The frozen
historical V1 dataset is not a current-time collector. To make a genuinely live
V1 prediction, callers must obtain current values and history that were
actually published by their claimed issue time; the API cannot authenticate
user-supplied provenance. The latest observed dispatch-down is a past input,
**not** the future value being predicted. If it is unavailable, this frozen
V1 model cannot be faithfully run from raw values without a different model.
`issue_timestamp_utc` is when the prediction is supposedly made;
`current_observation.available_at_utc` is when the **last required current
input actually became available**. If the latter is later, the response is
422: the model cannot use tomorrow's publication to make yesterday's
prediction. Do not simply type an earlier publication time to silence this
error. Use genuinely earlier data or a real later issue time. If you only want
a historical V1 example and lack the raw source history, use
`POST /predict/from-dataset` instead.

For a manual V2 prediction, inspect
`GET /model-info/daily-curtailment/raw-input-schema`, then send
`POST /predict/curtailment/from-raw` with a `target_date_utc` and 96 hourly
weather **forecast** rows: 24 hours each for west/Galway, south/Cork,
east/Dublin and north/Belfast. Each row needs its UTC target hour, forecast
publication time, 100 m wind speed (km/h), shortwave radiation (W/m²), and
2 m temperature (°C). The API builds the same daily aggregates and calendar
features used at training time; it does not accept realised future weather or
curtailment labels. Historical/current dates require a forecast claimed
available by 00:00 UTC on the target day. Future dates are limited to the next
seven UTC days and use the actual request time as issue time. Multi-day lead
accuracy is unvalidated. For either raw route, `input_provenance` says
`user_supplied_unverified`: shape/timing checks do not prove the forecast or
source publication was genuine.
If you lack 96 source forecasts, `POST /predict/curtailment/day` fetches an
archived daily forecast for one historical/current UTC day, while
`POST /predict/curtailment/window` replays 1–7 consecutive dates already in
the bundled V2 model dataset.

### Replay seven full UTC dataset days with V2

Use the same API key with `POST /predict/curtailment/window`. The coverage response above
provides `window_date_min_utc` and `window_date_max_utc`. For a seven-day window,
choose a start date whose seven consecutive days are all within that dataset:

```json
{"start_date_utc":"2026-04-28","days":7}
```

This returns an ordered `predictions` array for April 28–May 4. Each item is one
full-day curtailment probability and MWh estimate from the already stored weather
forecast/calendar features. Curtailment outcome columns are never fed to the model.
The API rejects any missing or future dataset date, and it makes no live-weather
request. This is historical replay, **not** the next seven days from now. Dates
used to fit the model are in-sample; the saved held-out evaluation remains the
honest performance measure. Check each `target_date_utc` at
`/actuals/daily-curtailment` if you want a separately reported observed outcome.

## Make a prediction from PowerShell

```powershell
$baseUrl = "https://YOUR-SERVICE.onrender.com"
$headers = @{ "X-API-Key" = "YOUR-TEAM-KEY" }

# Check that the model is ready.
Invoke-RestMethod -Uri "$baseUrl/health"

# Get both horizons at the latest bundled time.
$latest = Invoke-RestMethod `
  -Uri "$baseUrl/predict/latest?flexible_load_capacity_mw=100" `
  -Headers $headers
$latest.predictions | Format-Table

# Choose a valid historical time and request the 30-minute prediction.
$times = Invoke-RestMethod -Uri "$baseUrl/dataset/available-times?limit=5" -Headers $headers
$body = @{
  issue_timestamp_utc = $times.issue_timestamps_utc[-1]
  forecast_horizon_minutes = 30
  flexible_load_capacity_mw = 100
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "$baseUrl/predict/from-dataset" `
  -Headers $headers -ContentType "application/json" -Body $body
```

If no API key was configured, omit `-Headers $headers`.

## Make a prediction from Python

The example uses only the Python standard library:

```powershell
python examples/predict_from_api.py `
  --base-url https://YOUR-SERVICE.onrender.com `
  --api-key YOUR-TEAM-KEY `
  --capacity-mw 100
```

For a chosen historical time:

```powershell
python examples/predict_from_api.py `
  --base-url https://YOUR-SERVICE.onrender.com `
  --api-key YOUR-TEAM-KEY `
  --timestamp 2026-01-31T22:00:00Z `
  --horizon 30
```

For a two-hour prediction array:

```powershell
python examples/predict_from_api.py `
  --base-url https://YOUR-SERVICE.onrender.com `
  --api-key YOUR-TEAM-KEY `
  --timestamp 2026-01-15T12:00:00Z `
  --duration-hours 2 `
  --horizons 30 60
```

Frontend code can use `examples/frontend-prediction.js`. Do not put a valuable secret in public
browser code: anything shipped to a browser is visible to users. For a public website, keep the key
in the website's server-side environment and proxy requests to GridToEV.

## How to read a response

To retrieve the observed outcome later, copy `target_timestamp_utc` into
`GET /actuals/v1`. For a 24-hour prediction array, use `POST /actuals/v1/window`
starting at the first target, or send exact target timestamps to `POST /actuals/v1/batch`.
With both 30/60-minute horizons, a 24-hour issue window spans 24.5 hours of distinct
targets. Actual values can be `pending` until the EirGrid
archive is refreshed; see `docs/ACTUALS_API.md` for statuses and examples.

- `dispatch_down_probability`: probability of any dispatch-down in the target interval.
- `dispatch_down_event_prediction`: yes/no result after applying the learned threshold.
- `risk_level`: low, medium, or high display category.
- `predicted_dispatch_down_mwh`: central estimate of renewable energy not accepted by the grid.
- `predicted_curtailment_mwh` and `predicted_constraint_mwh`: components reconciled to the total.
- `prediction_interval_p10_mwh`, `p50`, and `p90`: low, middle, and high uncertainty estimates.
- `recoverable_surplus_mwh`: the smaller of predicted dispatch-down and the energy that the supplied
  flexible load could consume over the half-hour interval.

## Models in v1.1.0

The bundle contains seven fitted histogram gradient-boosting pipelines:

1. an event classifier for the probability of dispatch-down;
2. a dispatch-down residual regressor;
3. a curtailment regressor;
4. a network-constraint regressor;
5. a P10 dispatch-down quantile regressor;
6. a P50 dispatch-down quantile regressor; and
7. a P90 dispatch-down quantile regressor.

The central dispatch-down estimate also has a simple horizon-specific damped-trend guardrail. In the
current bundle, validation selected zero weight for the residual regressor at both horizons because it
was not consistently better in every rolling fold. In plain language, the API still loads all seven
models, but the stable recent-trend calculation currently supplies the central total; the classifier,
component models, and uncertainty models remain active.

## Stable-serving workflow

- Keep `release/shareable-prediction-api` deployed for colleagues.
- Develop and benchmark new data/features/models on separate feature branches.
- Replace the serving branch only after the new artifact passes the frozen benchmark and API tests.
- Roll back by redeploying the last known-good commit from the serving branch.

Never load a `.joblib` model from an untrusted source; its serialization format can execute code while
loading. Use the artifact committed in this repository or one produced by the trusted training job.
