# How GridToEV training and prediction work

## The short version

The first notebook prepares a clean table. The second notebook teaches several models from that table
and saves them as one model bundle. FastAPI loads that bundle when the server starts. The frontend
sends a time, forecast horizon, and flexible-load capacity; the API returns the likelihood and likely
amount of renewable electricity that will be dispatched down.

```text
Public energy files
        |
        v
01_build_model_ready_dataset.ipynb
        |
        v
gridtoev_model_ready.csv
        |
        v
02_train_and_export_models.ipynb
        |
        v
gridtoev_model_bundle.joblib
        |
        v
FastAPI  <---- HTTP request ---- Frontend
        |
        +---- JSON prediction ---> Frontend
```

## 1. Data preparation

Each dataset row represents a forecast made at `issue_timestamp_utc`. Its target is either 30 or 60
minutes later. The features describe conditions available at the issue time: wind, solar, demand,
prices, grid flows, SNSP, recent dispatch-down, ramps, and rolling history. Future dispatch-down values
are kept only in target columns.

## 2. Chronological model training

The code never shuffles the observations. It gives the oldest 70% of issue times to training, the next
15% to validation, and the newest 15% to testing. This imitates the real task: learn from the past and
predict the future.

Validation and expanding-window backtests choose the operating settings:

- the probability threshold that converts event probability into yes/no;
- a damped recent trend for each forecast horizon;
- whether the MWh machine-learning correction improves every historical backtest fold; and
- how much to widen the P10-P90 uncertainty range so it is better calibrated.

The final test period is not used for those choices.

## 3. Reproducible benchmark and final-test seal

`config/benchmark_contract.v1.json` freezes the split fractions, expanding-window folds, release
thresholds, dataset/model hashes, final-test dates, and expected v1.1.0 metrics. Run:

```powershell
python scripts/run_benchmark.py
```

The command first rejects unsorted rows, duplicate natural keys, incorrectly aligned targets, and any
forecast publication or source-vintage timestamp later than `issue_timestamp_utc`. It then fits on the
training partition and selects the event threshold, horizon-specific trend/blend settings, and interval
adjustment using only training and validation data. Only after those settings are frozen is the final
test passed to the scoring function.

The report includes overall, 30/60-minute, positive/no-event, and four rolling-fold results. The CSV
registry records the data hash, feature contract, model settings, runtime, rolling performance and final
metrics so future experiments can be compared on exactly the same contract. The benchmark runs in
memory and does not replace `models/gridtoev_model_bundle.joblib`.

## 4. Models in the bundle

The event classifier answers, “Is dispatch-down likely?” The central MWh forecast projects the latest
completed dispatch-down value with a damped trend learned separately for 30 and 60 minutes. A boosted-
tree residual model may adjust that forecast, but only when its adjustment improves every rolling
backtest fold. The curtailment and constraint models split the quantity into system-wide and network-
driven components. Three residual quantile models provide low, middle, and high estimates.

The learned components use histogram gradient-boosted trees. In plain language, they combine many
small decision trees that learn patterns such as “high wind, low demand, and low SNSP headroom usually
means greater risk.” The damped trend is deliberately simpler because it proved more stable on this
short dataset.

## 5. Saved training contract

The joblib file stores the fitted preprocessing and estimators together with the exact feature order.
Metadata stores the dataset hash, model version, package versions, test dates, probability threshold,
and blend weight. This stops the API from silently rearranging columns or loading an incompatible
feature set.

## 6. FastAPI serving

The server loads the model bundle once during startup. A request does not rerun either notebook and
does not retrain anything. Prediction is therefore quick enough for an interactive frontend.

The demo endpoints are:

- `GET /predict/latest`: predict both horizons for the latest shared dataset time.
- `POST /predict/from-dataset`: predict a selected historical issue time and horizon.
- `GET /dataset/available-times`: list times the frontend can offer in a picker.
- `GET /dataset/info`: explain the valid date range and exact UTC timestamp format.
- `POST /predict/window/from-dataset`: replay 30/60-minute predictions every half-hour over a
  selected 0.5-24 hour historical window and return an ordered array.

The live integration endpoint is:

- `POST /predict/features`: accept a complete 119-feature snapshot created by a future live data
  ingestion service.

Other useful endpoints are `GET /health`, `GET /model-info`, and the automatic interactive API page at
`/docs`.

The window endpoint is deliberately labelled `historical_rolling_short_horizon`. For example, a
24-hour request runs the existing 30/60-minute models at each half-hour whose feature row already
exists in the historical dataset. It does not turn the current model into a true day-ahead model.
Training real 2-hour and 24-hour horizons requires causally available forecast features and separate
horizon evaluation.

On a shared deployment, setting `GRIDTOEV_API_KEY` protects model information, dataset times, and all
prediction routes with an `X-API-Key` header. The root page and health check stay public so people and
the hosting platform can confirm that the service is running. See `SHARING_AND_PREDICTIONS.md` for the
team workflow and copyable request examples.

## 7. Prediction response

One response includes event probability, yes/no event classification, risk level, total dispatch-down
MWh, curtailment MWh, constraint MWh, P10/P50/P90 estimates, and recoverable energy under the supplied
flexible-load capacity. Curtailment and constraint predictions are proportionally reconciled so they add
up to the total estimate.

For a 100 MW flexible load and a 30-minute interval, at most 50 MWh can be redirected. The API returns
the lower of that 50 MWh capacity and the predicted dispatched-down energy.

## 8. Frontend connection

The browser calls the API with ordinary JSON. The API validates the request and returns ordinary JSON,
so the frontend can be React, Next.js, Vue, plain JavaScript, or a mobile application. CORS is enabled
for local development on ports 3000 and 5173 by default.

For a hackathon demo, use the dataset endpoints. For a live system, build a scheduled data adapter that
collects the newest EirGrid, ENTSO-E, SEMO, and weather observations, applies the same feature logic,
then calls `/predict/features`.

## 9. Current limitation

The combined dataset covers one fully populated month because the organiser price sample is limited to
January 2026. The improved MWh forecast beats the older, one-interval-stale persistence baseline on the
held-out period, but the rolling guardrail gives the boosted-tree residual correction zero weight until
it proves stable across regimes. The test period contains no positive curtailment examples. Expanding
the training history remains the next modelling priority.
