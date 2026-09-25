# Capacity-weighted regional weather signals

Issue #7 adds regional wind and solar weather context without creating a
separate model for each region. The deployed modelling shape remains one row
per national issue time and forecast horizon. Regional forecasts are retained
as features and combined with separate wind- and solar-capacity weights.

## Source choice and licence

Met Éireann is the preferred national source. Its open-data catalogue provides
current point forecasts, observations, and historical datasets, but it does not
offer a convenient archive of the exact January 2026 point-forecast vintages
needed by this benchmark.

The implementation therefore uses the documented fallback:

- Open-Meteo Single Runs API with ECMWF IFS HRES archives for forecast
  vintages;
- Open-Meteo Historical Forecast API as a delayed analysis proxy for causal
  forecast-error features.

The raw API responses are Open-Meteo/ECMWF data under CC BY 4.0. Attribution is
required. GridToEV modifies the data through half-hour interpolation, regional
capacity weighting, vector conversion, rolling error calculations, and feature
engineering. Raw responses are cached below `data/raw/weather/` and excluded
from Git. URLs, first retrieval times, checksums, schema versions, and row
coverage are committed in `data/processed/regional_weather_source_manifest.csv`.

## Forecast-vintage semantics

Every archived run has three different times:

- `run_initialized_at_utc`: the model's reference or initialisation time;
- `published_at_utc`: when GridToEV conservatively considers the output usable;
- `valid_timestamp_utc`: the hour the weather prediction describes.

An ECMWF run is not public at its initialisation instant because it must still
be computed and distributed. Open-Meteo documents a typical four-to-six-hour
delay for global models. GridToEV therefore sets publication time to run time
plus six hours. A model row can select a weather value only when
`published_at_utc <= issue_timestamp_utc`.

Runs are collected every six hours and retain fourteen hourly steps. Hourly
values are interpolated to the half-hour grid only within the same run. This
allows 30- and 60-minute targets to use one coherent, already-published
forecast; adjacent runs are never blended.

The historical-forecast series is not labelled as a station observation. It is
an `analysis_proxy`. To prevent it from becoming future information, each proxy
value is considered observable only six hours after its valid time. Rolling
weather errors include only proxy values whose availability time is no later
than the model issue time.

## Regions and capacity weights

The six representative points and exact inputs are versioned in
`config/regional_weather_capacity_weights.csv`:

| Region | Representative point | Wind proxy MW | Solar proxy MW |
| --- | --- | ---: | ---: |
| Northwest | 54.80, -7.80 | 1,100 | 90 |
| West | 53.35, -9.05 | 900 | 120 |
| Southwest | 52.20, -9.70 | 1,400 | 250 |
| Midlands | 53.20, -7.70 | 600 | 400 |
| East | 53.35, -6.30 | 300 | 450 |
| Southeast | 52.30, -6.50 | 700 | 450 |

The wind values sum to the approximately 5 GW national onshore fleet and the
solar values to the approximately 1.76 GW 2025 fleet. Their regional allocation
is a transparent engineering proxy informed by the SEAI county dashboard, not
an official asset register. The builder validates non-negative capacities and
normalizes wind and solar independently to sum to one. Updating the CSV changes
the weights reproducibly without changing feature code.

Each region has an explicit missing flag. Weighted means divide by the
available regional weight, while coverage ratios state how much of the total
capacity proxy was represented. Missing weather is never replaced by calm wind,
clear sky, or a national average.

## Engineered features

Forecast features include:

- 100-m wind speed and meteorological direction converted to `u` and `v`
  components;
- 10-m gust, gust excess over 100-m wind speed, and cross-region spatial
  spread;
- three-hour mean-sea-level-pressure tendency;
- 2-m temperature, total cloud cover, and shortwave irradiance;
- wind-capacity-weighted and solar-capacity-weighted national summaries;
- the regional wind, gust, temperature, cloud, and irradiance values;
- forecast age, source availability, per-region missing flags, and wind/solar
  coverage ratios;
- latest causal wind error and 24-hour wind bias/MAE;
- 24-hour temperature bias and irradiance bias/MAE.

## Quality and coverage

The committed build contains:

- 120 ECMWF model-run vintages;
- 10,080 regional forecast-vintage rows;
- 4,608 delayed analysis-proxy rows;
- six representative regions;
- 2,867 feature rows and 65 columns;
- 100% weather-forecast and regional-capacity coverage;
- 99.16% causal-error coverage, with the initial rows correctly missing because
  six hours of prior proxy availability did not yet exist;
- zero duplicate keys, horizon errors, future-publication selections, and
  infinite numeric values.

## Development ablation

The benchmark evaluates six variants on identical development folds:

- all regional weather features;
- aggregate forecasts plus causal errors;
- aggregate forecasts only;
- causal errors only;
- wind aggregates only;
- solar aggregates only.

Three expanding folds inside the 70% training period and the following 15%
validation period are used. The final 15% test period remains sealed.

The best development variant was `causal_errors_only`, containing eight
features. Its mean per-fold improvement was effectively zero (`0.006%`), its
worst fold degraded by `2.19%`, and its unweighted mean MAE changed from 12.547
MWh to 12.606 MWh. It does not clear the release rule of at least 15% mean
per-fold improvement with no degrading fold. Weather features are therefore
**excluded from the current production model** but remain ready for a longer
multi-season training window and improved regional capacity registry.

## Rebuild and outputs

From the repository root:

```powershell
$env:PYTHONPATH = "src"
python scripts/build_regional_weather_data.py --workers 4
```

The command is resumable. Existing raw responses are checksum-verified and not
downloaded again.

Outputs:

- `data/processed/regional_weather_forecast_vintages.csv.gz`
- `data/processed/regional_weather_analysis_proxy.csv.gz`
- `data/processed/regional_weather_features_30_60.csv`
- `data/processed/regional_weather_feature_dictionary.csv`
- `data/processed/regional_weather_source_manifest.csv`
- `data/processed/regional_weather_quality_report.json`
- `benchmarks/regional_weather_signals/ablation_report.json`
- `benchmarks/regional_weather_signals/experiment_registry.csv`
