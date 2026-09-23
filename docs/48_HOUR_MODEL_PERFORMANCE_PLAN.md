# GridToEV 48-hour model-performance sprint

## Outcome

Within 48 hours, build and evaluate a release candidate that predicts Irish renewable dispatch-down
30 and 60 minutes ahead more accurately and robustly than model version 1.1.0. The sprint focuses on
the dispatch-down MWh forecast because the event classifier is already strong and the present one-month
training window is the main constraint.

This is an evidence-based performance target, not a guarantee. If a candidate does not pass every
release guardrail, the current v1.1.0 bundle remains the production fallback.

## Starting point

| Measure | Current held-out result | 48-hour release target | Stretch target |
| --- | ---: | ---: | ---: |
| Dispatch-down MAE | 11.83 MWh | <= 9.50 MWh | <= 8.50 MWh |
| Dispatch-down WAPE | 17.22% | <= 14.0% | <= 12.5% |
| Dispatch-down RMSE | 20.59 MWh | <= 18.0 MWh | <= 16.5 MWh |
| P50 pinball loss | 7.00 | <= 6.0 | <= 5.5 |
| P10-P90 empirical coverage | 74.48% | 78%-90% | 80%-88% |
| Event average precision | 0.9998 | >= 0.995 | maintain current |

The original deployed MAE was 18.85 MWh. Reaching 9.50 MWh would make the total improvement from that
starting point approximately 50%, while improving the current v1.1.0 result by about 20%.

## Non-negotiable release guardrails

1. Evaluate chronologically. Never use a random train/test split.
2. Keep one final contiguous test period untouched until model selection is complete.
3. Require the winning approach to improve aggregate rolling-origin MAE by at least 15% and avoid a
   material regression in either the 30- or 60-minute horizon.
4. Compare every candidate with latest-observation, stale-persistence, and v1.1.0 baselines.
5. Use a feature only when its value was available at `issue_timestamp_utc`. Store
   `published_at_utc`, `source_vintage_utc`, and `downloaded_at_utc` where the source supports them.
6. Reject forecast revisions published after the issue time, and use actual imbalance, price, or
   dispatch-down values only after their interval was complete and published.
7. Preserve a reproducible source manifest containing URLs, hashes, coverage, retrieval times, and
   licensing notes. Raw source files remain organised by provider, report, and year.
8. Keep v1.1.0 as the rollback artifact until the new bundle passes dataset, model, API, and notebook
   checks.

## Public data priorities

### P0: history and forecast vintages

| Source | Data to collect | Highest-value variables |
| --- | --- | --- |
| [EirGrid System and Renewable Data Reports](https://www.eirgrid.ie/grid/system-and-renewable-data-reports) | Historical system, renewable, and dispatch-down reports | wind/solar generation, demand, fuel mix, SNSP, dispatch-down, curtailment, constraint |
| [EirGrid wind dashboard](https://www.smartgriddashboard.com/all/wind/) | 15-minute actual and forecast wind | issued forecast, actual, forecast error, forecast revision, rolling bias |
| [EirGrid solar dashboard](https://www.smartgriddashboard.com/all/solar/) | 15-minute actual and forecast solar | issued forecast, actual, forecast error, renewable share |
| [EirGrid demand dashboard](https://www.smartgriddashboard.com/all/demand/) | 15-minute actual and forecast demand | issued forecast, actual, forecast error, forecast net load |
| [EirGrid interconnection dashboard](https://www.smartgriddashboard.com/all/interconnection/) | EWIC, Moyle, and Greenlink flows | import/export flow, utilisation, ramp, export headroom |

The minimum useful history is 24 months; the preferred range is 2021 to the latest available complete
month. The January 2026 price sample must not continue to restrict the entire table to one month. When
one optional source has shorter coverage, preserve the longer core dataset and add source-availability
flags rather than dropping old rows.

### P1: market, operational, and network state

| Source | Reports or data | Candidate features |
| --- | --- | --- |
| [SEMO static reports](https://www.sem-o.com/market-data/static-reports) | `PUB_15MinAggWindFcst`, `PUB_4DayAggRollWindUnitFcst`, `PUB_DailyLoadFcst`, `PUB_RTCOperationalSchedule`, `PUB_HlfHrlyAggregatedPN`, `PUB_HrlyForecastImbalance`, `PUB_HrlyNetImbalVolumeForecast`, `PUB_DailyIntconNTC`, `PUB_5MinImbalPrc`, `PUB_30MinAvgImbalPrc`, `PUB_HrlySsiiSiff` | wind/demand forecast revisions, scheduled-minus-actual output, downward margin, imbalance forecast, NTC, export headroom, lagged imbalance price/volume |
| [EirGrid outage information](https://www.eirgrid.ie/industry/customer-information/outage-information) | Weekly all-island outage plan and transmission outage programme | scheduled/forced unavailable MW, outage counts, largest unavailable unit, regional outage flags |
| [EirGrid ECP constraint forecasts](https://www.eirgrid.ie/industry/customer-information/ecp-constraint-forecast-reports) | Constraint group reports | group headroom, binding-risk flags, regional constraint pressure |

### P2: weather and wider-system enrichment

| Source | Data | Candidate features |
| --- | --- | --- |
| [Met Eireann open data](https://www.met.ie/about-us/specialised-services/open-data) | Observations, forecasts, and reanalysis | capacity-weighted wind speed/direction, gust, pressure, temperature, cloud and irradiance |
| [ENTSO-E Transparency API](https://transparency.entsoe.eu/content/static_content/download?path=/Static+content/web+api/IG-for-TP-data-extraction-process.pdf) | Load, generation, transmission, balancing, and outage data | cross-checks, neighbouring-system stress, transfer capacity |
| [Open-Meteo historical forecast API](https://open-meteo.com/en/docs/historical-forecast-api) | Archived operational forecast model runs | fallback historical weather vintages when first-party forecast archives are insufficient |

ENTSO-E may require an access token, and each source's redistribution terms must be recorded before raw
files are committed or republished.

## Target feature families

- Forecast state: wind, solar, demand, net load, renewable share, renewable surplus and SNSP proxy at
  30 and 60 minutes.
- Forecast quality: latest forecast error, 3/6/12/24-hour rolling error, rolling bias, forecast
  revision and disagreement between forecast vintages.
- Flexibility: interconnector NTC, scheduled flow, export headroom, utilisation and ramps.
- Operational margin: scheduled generation, physical notifications, downward margin, imbalance
  forecast, lagged net imbalance volume and lagged imbalance price.
- Network availability: forced and planned generation outages, transmission outages, affected
  constraint groups and time-to-outage-end.
- Weather regime: capacity-weighted regional weather, wind direction components, gust spread,
  pressure tendency, irradiance and weather forecast errors.
- Regime context: hour/day/season, holidays, recent dispatch-down persistence, ramps and duration of
  the current event.

## Organised dataset layout

```text
data/
  raw/
    eirgrid/system/<year>/
    eirgrid/forecasts/<series>/<year>/
    eirgrid/outages/<year>/
    semo/<report>/<year>/
    weather/met_eireann/<year>/
    weather/open_meteo/<year>/
  interim/
    source_aligned/
    forecast_vintages/
  processed/
    gridtoev_model_ready.csv
    gridtoev_data_dictionary.csv
    gridtoev_quality_report.json
    source_manifest.csv
```

Large raw files remain outside Git. Download scripts and the manifest must be sufficient to reproduce
them. Interim files may use Parquet for speed; the final modelling table remains CSV-compatible for the
hackathon workflow.

## 48-hour execution schedule

| Window | Work | Exit condition |
| --- | --- | --- |
| Hours 0-3 | Freeze v1.1.0 benchmark, rolling folds, data-availability contract, and experiment log | Reproducible baseline report with per-horizon and per-fold metrics |
| Hours 2-12 | Download and normalise the longest feasible EirGrid system and dispatch-down history | At least 24 months preferred; explicit coverage report if the source offers less |
| Hours 3-14 | Ingest wind, solar, demand, and interconnector forecast vintages | As-of join tests prove every forecast existed by issue time |
| Hours 8-18 | Ingest SEMO operational, imbalance, schedule, PN, and NTC reports | Normalised half-hour table with publication-lag flags |
| Hours 10-20 | Add outage/constraint signals; start regional weather if P0 sources are stable | Source-specific quality and coverage checks pass |
| Hours 14-26 | Build forecast-error, surplus, headroom, operational-margin, outage and weather features | No duplicate natural keys; targets reconcile; leakage audit passes |
| Hours 22-34 | Benchmark horizon-specific linear, tree, boosting and two-stage candidates | Experiment table compares all candidates on identical rolling folds |
| Hours 30-39 | Run ablations, tune stable ensembles, calibrate quantiles, inspect event/regime errors | Winning feature groups demonstrate repeatable gain, not one-fold luck |
| Hours 38-45 | Rebuild both notebooks, saved pipeline, metadata, and FastAPI contract | End-to-end training and API tests pass with the winning bundle |
| Hours 45-48 | Execute untouched final test, document result, package release candidate and rollback | Release gate passes, or v1.1.0 remains active with findings recorded |

## Model experiment ladder

Run the inexpensive, interpretable candidates first and stop spending time on approaches that cannot
beat persistence across folds.

1. Current damped trend and persistence baselines, separately by horizon.
2. Regularised linear/Huber residual models using forecast deltas and operational margins.
3. Scikit-learn histogram gradient boosting and extra-trees residual models.
4. Optional LightGBM, XGBoost, or CatBoost only if installation is reliable and the same folds show a
   clear gain over the standard-library candidates.
5. Two-stage event-probability times positive-quantity models for zero-inflated targets.
6. Horizon-specific ensembles whose weights are selected from out-of-fold predictions and accepted
   only if each horizon remains stable.
7. Quantile or conformal calibration fitted on validation predictions, never on the final test period.

Record MAE, RMSE, WAPE, pinball loss, coverage, event average precision, training time, inference time,
feature count, fold dispersion and worst-fold degradation. Produce ablations for forecast-error,
operational, outage and weather groups so the team knows which public data actually helped.

## GitHub issue backlog and dependency order

Tracking epic: [#11 - 48-hour dispatch-down model performance sprint](https://github.com/Carlson29/GridToEv/issues/11).

1. [#1 - **[P0] Freeze leakage-safe rolling benchmark and experiment registry**](https://github.com/Carlson29/GridToEv/issues/1)
2. [#2 - **[P0] Extend EirGrid dispatch-down and system history to 2021-present**](https://github.com/Carlson29/GridToEv/issues/2)
3. [#3 - **[P0] Ingest wind, solar, demand, and interconnector forecast vintages**](https://github.com/Carlson29/GridToEv/issues/3)
4. [#4 - **[P0] Engineer forecast-error, renewable-surplus, and grid-headroom features**](https://github.com/Carlson29/GridToEv/issues/4)
5. [#5 - **[P1] Add SEMO imbalance, operational schedule, PN, and NTC signals**](https://github.com/Carlson29/GridToEv/issues/5)
6. [#6 - **[P1] Add generation outages, transmission outages, and constraint-group signals**](https://github.com/Carlson29/GridToEv/issues/6)
7. [#7 - **[P2] Add capacity-weighted regional weather forecast features**](https://github.com/Carlson29/GridToEv/issues/7)
8. [#8 - **[P0] Benchmark horizon-specific models and stable ensembles**](https://github.com/Carlson29/GridToEv/issues/8)
9. [#9 - **[P1] Calibrate uncertainty intervals and component forecasts**](https://github.com/Carlson29/GridToEv/issues/9)
10. [#10 - **[P0] Integrate the winning pipeline into notebooks, FastAPI, and release checks**](https://github.com/Carlson29/GridToEv/issues/10)

```text
Benchmark (1) ------------------------------+--------------------+
History (2) --------+                       |                    |
Forecasts (3) ------+--> feature build (4) -+--> models (8) -----+--> release (10)
SEMO (5) -----------+                       |          |         |
Outages (6) --------+                       |          +--> calibration (9)
Weather (7, stretch)+-----------------------+
```

Issues 1-4, 8, and 10 are the critical path. Issues 5-7 are independently testable enrichments; they
enter the final model only when an ablation proves value. Issue 9 must use out-of-fold predictions from
issue 8. Issue 10 may begin with contract and test changes before model selection finishes.

## Issue specifications

### 1. Freeze leakage-safe rolling benchmark and experiment registry

**Timebox:** 3 hours. **Depends on:** nothing. **Blocks:** every model comparison.

- Freeze the current v1.1.0 artifact, data hash, test dates and metrics as the rollback benchmark.
- Define expanding-window folds shared by every experiment and report aggregate, per-horizon,
  per-fold and dispatch-down-event-regime metrics.
- Add a machine-readable experiment table with data version, feature groups, model settings, runtime
  and metrics.
- Add tests proving train/validation/test time ranges do not overlap and model selection never reads
  the final test labels.

**Accept when:** one command reproduces the 11.83 MWh baseline; fold definitions and release thresholds
are versioned; an intentionally shuffled or future-leaking dataset fails a test.

### 2. Extend EirGrid dispatch-down and system history to 2021-present

**Timebox:** 10 hours. **Depends on:** issue 1's time contract. **Blocks:** issues 4 and 8.

- Build resumable downloads and parsers for EirGrid historical system, renewable and dispatch-down
  reports, partitioned by source/report/year.
- Normalise timestamps to UTC and half-hour settlement intervals while retaining original timestamps,
  timezone/DST information and provenance.
- Keep the long core history even where price or optional sources are missing; add availability flags.
- Extend the manifest with URL, retrieval timestamp, hash, licence note, first/last interval, row count
  and schema version.

**Accept when:** the core table covers at least 24 months when the upstream archive permits; duplicate
keys are zero; dispatch-down equals curtailment plus constraint within documented tolerances; rerunning
the downloader does not redownload unchanged files.

### 3. Ingest wind, solar, demand, and interconnector forecast vintages

**Timebox:** 11 hours. **Depends on:** issue 1's availability contract. **Blocks:** issue 4.

- Collect EirGrid/SEMO wind, solar and demand forecasts plus EWIC, Moyle and Greenlink flows or
  schedules at the finest reliable cadence.
- Retain forecast target time, issue/publication time, revision/vintage, downloaded time and source.
- Implement backward as-of joins that select only the latest forecast published by each model issue
  timestamp.
- Add coverage, staleness and source-availability flags instead of silently filling long gaps.

**Accept when:** 30- and 60-minute forecast values can be reproduced for every covered issue time; no
selected vintage has `published_at_utc > issue_timestamp_utc`; DST, duplicate-revision and late-file
tests pass.

### 4. Engineer forecast-error, renewable-surplus, and grid-headroom features

**Timebox:** 8 hours. **Depends on:** issues 2 and 3; may also consume issues 5-7. **Blocks:** issue 8.

- Create forecast net load, renewable share, renewable surplus, SNSP proxy, export headroom and
  interconnector-utilisation features for each horizon.
- Create causal wind/solar/demand forecast errors, revisions, rolling MAE/bias and disagreement
  features using only completed intervals and known vintages.
- Add explicit missingness/staleness indicators and update the data dictionary with formula, unit,
  source and availability rule.
- Add invariants for horizon alignment, rolling-window endpoints and finite model inputs.

**Accept when:** every feature has a documented causal availability rule; leakage tests fail when a
future observation or revision is injected; the clean table has unique natural keys and passes all
existing quality checks.

### 5. Add SEMO imbalance, operational schedule, PN, and NTC signals

**Timebox:** 10 hours. **Depends on:** issue 1. **Feeds:** issues 4 and 8.

- Ingest the relevant SEMO static reports for aggregate wind/load forecasts, RTC schedules, aggregated
  physical notifications, forecast imbalance, net imbalance volume forecast, interconnector NTC,
  imbalance price and system shortfall indices.
- Standardise units, settlement periods, publication times, revisions and Ireland/Northern Ireland
  scope.
- Engineer scheduled-minus-actual generation, downward margin, imbalance forecast, export headroom,
  and lagged actual imbalance/price signals.

**Accept when:** parsers are fixture-tested against schema variants; actual market outcomes are lagged
until available; the feature-family ablation is recorded even if the group is rejected.

### 6. Add generation outages, transmission outages, and constraint-group signals

**Timebox:** 8 hours. **Depends on:** issue 1. **Feeds:** issues 4 and 8.

- Parse the EirGrid all-island outage plan, transmission outage programme and ECP constraint reports.
- Separate planned from forced outages and derive unavailable MW, count, largest unavailable unit,
  regional/constraint-group pressure and time-to-start/end.
- Preserve publication snapshots so planned outages are represented as they were known at issue time.

**Accept when:** features are causal, unit-tested across overlapping outages, joined without duplicate
model rows, and included in a named ablation with coverage reported.

### 7. Add capacity-weighted regional weather forecast features

**Timebox:** 8 hours and explicitly stretch scope. **Depends on:** issue 1. **Feeds:** issues 4 and 8.

- Obtain Met Eireann forecast/observation history; use Open-Meteo historical forecast runs only as a
  documented fallback.
- Map weather points to representative wind/solar regions and capacity weights without storing
  personal or restricted data.
- Engineer wind-speed and direction components, gust/spread, pressure tendency, temperature, cloud,
  irradiance and causal forecast-error features.

**Accept when:** source licence and vintage semantics are documented; coverage and station/grid weights
are reproducible; the weather-only ablation improves a rolling metric or is excluded from production.

### 8. Benchmark horizon-specific models and stable ensembles

**Timebox:** 12 hours. **Depends on:** issues 1 and 4; consumes completed 5-7. **Blocks:** issues 9 and 10.

- Compare the current baseline with Huber/regularised linear residuals, histogram gradient boosting,
  extra trees and a two-stage event-times-positive-quantity approach.
- Tune 30- and 60-minute models independently using identical rolling predictions; add optional
  LightGBM/XGBoost/CatBoost only after standard candidates run reliably.
- Select ensemble weights from out-of-fold predictions and reject candidates that improve only one
  lucky period or materially regress one horizon.
- Publish ablations, error slices and inference/training cost alongside metrics.

**Accept when:** the experiment table is reproducible; the winner beats v1.1.0 aggregate rolling MAE by
at least 15%, avoids material per-horizon regression and targets final-test MAE <= 9.50 MWh; otherwise
v1.1.0 is retained and the negative result is documented.

### 9. Calibrate uncertainty intervals and component forecasts

**Timebox:** 6 hours. **Depends on:** issue 8's out-of-fold predictions. **Feeds:** issue 10.

- Calibrate P10/P50/P90 forecasts without touching final-test labels and enforce non-crossing quantiles.
- Compare horizon-specific conformal adjustments and report interval coverage plus average width.
- Revisit curtailment/constraint component models only after longer history supplies positive examples;
  reconcile component predictions exactly to the total.

**Accept when:** P50 pinball loss is <= 6.0, P10-P90 coverage is 78%-90% without excessive width,
quantiles never cross, components are non-negative and reconcile to total MWh.

### 10. Integrate the winning pipeline into notebooks, FastAPI, and release checks

**Timebox:** 8 hours. **Depends on:** issues 2-4 and 8; issue 9 for calibrated intervals.

- Make both notebooks run top-to-bottom from a clean environment with concise explanations and no
  hidden state.
- Export a versioned preprocessing/model bundle and metadata containing features, hashes, folds,
  settings, source coverage and library versions.
- Update FastAPI validation and model-info output for the new feature contract while preserving both
  forecast horizons and frontend response fields.
- Add end-to-end tests for build, train, save, load, predict, schema errors and rollback.

**Accept when:** clean notebook execution succeeds; all tests pass; the API serves the release candidate
with measured metrics and can load v1.1.0 as rollback; documentation states whether the release gate
passed and identifies unavailable or rejected sources.

## Definition of done

- The combined table covers at least 24 months when upstream availability permits and reports exact
  coverage for every source and feature family.
- Every row is unique on `issue_timestamp_utc` plus `forecast_horizon_minutes`, target timestamps are
  strictly later than issue timestamps, and required model columns contain no unhandled missing values.
- Automated tests fail on future-published forecast vintages, noncausal rolling windows, duplicate
  keys, target leakage, schema drift, or target-accounting errors.
- A versioned experiment report shows identical rolling folds and ablations for every candidate.
- The selected candidate meets the release guardrails and target metrics; otherwise the report clearly
  states that no replacement was promoted.
- Both Jupyter notebooks execute from a clean environment and explain each data source, transformation,
  leakage rule, model decision and limitation.
- The saved bundle contains preprocessing, feature order, source/data hashes, model settings, calibration
  values and dependency versions.
- FastAPI loads the new bundle, validates the feature contract, serves both horizons, and passes all
  unit and end-to-end tests.

## Fast decisions when time is tight

- Prefer more chronological history and true forecast vintages over adding many same-time actuals.
- Do not block the core build on weather, ENTSO-E access, or one hard-to-parse report.
- Preserve rows with source-availability flags rather than shrinking the shared window to the shortest
  optional source.
- Drop a source after a two-hour parsing spike if it has no usable publication timestamp or cannot be
  joined causally; record the reason and move to the next source.
- Promote only measured gains. A more complex model that loses to the damped-trend baseline stays out of
  production.
