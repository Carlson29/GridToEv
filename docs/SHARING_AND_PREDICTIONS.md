# Share the GridToEV model and request predictions

## What colleagues receive

The `release/shareable-prediction-api` branch is the stable serving branch. It contains the frozen
v1.1.0 model bundle, the small demonstration dataset, FastAPI, Docker packaging, and request examples.
The API loads the bundle once at startup; a prediction request does not run a notebook and does not
retrain the model. Model experiments can therefore continue on a different branch without changing
the predictions used by colleagues.

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
6. Deploy, wait until `/health` returns `{"status":"ready",...}`, then share only these two values
   through a private team channel:

   - the Render URL, such as `https://gridtoev-api.onrender.com`;
   - the API key.

Do not commit the key. A free service may sleep when idle, so its first request can be slower. Render
uses `/health` to reject a deployment that cannot load the model.

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
5. For an array covering 2, 24, or up to 48 hours of historical issue times, call
   `POST /predict/window/from-dataset`, supply a valid `start_timestamp_utc`, and use
   `duration_hours: 2`, `24`, or `48`.

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
times, or eight prediction objects when both horizons are selected. A 48-hour request returns 96 issue
times, or 192 prediction objects with both horizons. The response also includes a separate summary
for each horizon so totals are not accidentally double-counted.

This is a rolling historical replay: every item is still a 30- or 60-minute forecast using the feature
row available at that item's issue time. It is useful for charts, demonstrations, and backtesting, but
it is not a single forecast made 2, 24, or 48 hours ahead.

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
