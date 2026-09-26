# Forecast-error and grid-headroom features

This is the implementation guide for Issue #4. The pipeline converts the
forecast-vintage archive and completed EirGrid observations into one row per
`issue_timestamp_utc` and forecast horizon. It supports the 30- and 60-minute
horizons already used by GridToEV.

## Build the artifacts

From the repository root, run:

```powershell
python scripts/build_forecast_features.py
```

The command uses the normalized Issue #2 and #3 outputs and writes:

| Artifact | Purpose |
|---|---|
| `data/processed/forecast_model_features_30_60.csv` | Combined modelling and audit table. |
| `data/processed/forecast_model_feature_dictionary.csv` | Formula, unit, source, role and availability rule for every column. |
| `data/processed/forecast_model_feature_quality_report.json` | Coverage, missingness, key, horizon and leakage checks. |

`scripts/build_extended_data.py` also creates these files after refreshing the
history and forecast archives. The equivalent commented workflow is in
`notebooks/04_engineer_causal_forecast_features.ipynb`.

## Features and their meaning

- `forecast_net_load_mw` is forecast demand minus forecast wind and solar. It
  approximates how much demand must be served by conventional generation and
  imports.
- `forecast_renewable_share_ratio` and
  `forecast_renewable_surplus_mw` describe renewable penetration and the amount
  by which forecast wind plus solar exceeds forecast demand.
- `forecast_snsp_proxy_ratio` adds positive completed net interconnector flow
  to forecast wind and solar, then divides by demand. It is a transparent proxy,
  not the official operational SNSP calculation.
- Per-interconnector and total `export_headroom` use the conservative formula
  `max(export NTC - abs(latest completed flow), 0)`. Utilisation uses absolute
  completed flow divided by the larger of import and export NTC.
- Demand, wind and solar revision features report the latest change, known
  vintage count, standard deviation and range at the issue time.
- Forecast error is `actual - forecast`. A positive value means the forecast
  was too low. The table includes the latest completed error and causal rolling
  MAE, bias and observation count over 2, 6, 12 and 24 hours.

## Availability and leakage controls

The issue time is the only information cut-off. A forecast publication is
eligible only when `published_at_utc <= issue_timestamp_utc`. A half-hour actual
at time `t` is considered completed at `t + 30 minutes`; error and system-state
features require that completion time to be no later than the issue time.

The natural key is `issue_timestamp_utc + forecast_horizon_minutes`, and the
pipeline verifies that target time equals issue time plus the horizon. It fails
closed on duplicate keys, future publications, future observations, horizon
misalignment or infinite numeric values. Tests deliberately inject future
forecast and observation timestamps to prove those guards work.

Forecast staleness thresholds are 2 hours for wind, 6 hours for demand and 24
hours for solar and interconnector capacity. Completed system state becomes
stale after 90 minutes. Stale completed flows are not used to calculate current
headroom, utilisation or the SNSP proxy.

## Missing data and the current coverage gap

Missing values are not silently converted to zero. Nullable numeric and
timestamp fields receive a paired `_missing_flag`; source forecast fields also
retain `_available_flag`, and age-sensitive inputs receive `_stale_flag`. The
fitted scikit-learn pipelines use median imputation with missing indicators for
numeric model inputs.

The current retained forecast archive begins after the latest published
half-hour actual-history file. Consequently, the generated live table has no
honest forecast-error values or target labels yet, and its completed state is
stale. The quality report states this directly. These columns will populate
automatically when a refresh supplies overlapping forecasts and actuals; the
pipeline does not fabricate historical vintages or carry stale grid state into
headroom features.

## Sign conventions

The existing EirGrid interconnector flow sign is treated as positive net import
for the SNSP proxy. Headroom and utilisation deliberately use absolute flow so
their conservative capacity calculation is not sensitive to direction. This
assumption is recorded in the data dictionary and should be rechecked if the
publisher changes its schema or sign convention.
