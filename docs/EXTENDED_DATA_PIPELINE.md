# Extended history and forecast-vintage pipeline

This pipeline completes performance-backlog Issues #2, #3 and #4. It fixes the most important data
weakness in the January-only baseline: too little history, and no auditable record of which forecast
revision was available when a prediction would have been made. It also builds causal forecast-error,
renewable-surplus and grid-headroom features.

## Run it

From the repository root:

```powershell
python scripts/build_extended_data.py --forecast-days 3
```

`--forecast-days` controls the requested SEMO archive window. The upstream API still decides how many
publications it retains. `--workers` controls concurrent public-file downloads and defaults to eight.
Raw files are never overwritten; a rerun verifies the cached checksum and skips the network request.

The same workflow, quality summaries, and inspection cells are in
`notebooks/03_extend_history_and_forecast_vintages.ipynb`.

## What is produced

| Artifact | Grain and purpose |
|---|---|
| `data/processed/eirgrid_core_history_30min.csv.gz` | One UTC half-hour. System state, named interconnector flows, dispatch-down labels, and field-level availability flags. |
| `data/processed/forecast_vintages.csv.gz` | One forecast feature, target interval, and publication revision. Preserves publication, retrieval, source, region, value, and original time strings. |
| `data/processed/forecast_features_asof_30_60.csv` | One issue time and 30/60-minute target. Contains only the latest revision published no later than that issue time. |
| `data/processed/forecast_model_features_30_60.csv` | Issue #4 combined features: forecast state, revisions, causal errors and interconnector headroom. |
| `data/processed/forecast_model_feature_dictionary.csv` | Formula, unit, source and availability rule for every engineered-table column. |
| `data/processed/forecast_model_feature_quality_report.json` | Natural-key, horizon, leakage, missingness and staleness checks. |
| `data/processed/extended_source_manifest.csv` | URL, local path, provider/report/year, retrieval time, checksum, licence note, schema version, row count, and first/last interval. |
| `data/processed/*quality_report.json` | Coverage, duplicate-key checks, accounting tolerance, publication lag distributions, feature availability, and leakage checks. |

Pandas reads the gzip tables directly:

```python
history = pd.read_csv("data/processed/eirgrid_core_history_30min.csv.gz")
vintages = pd.read_csv("data/processed/forecast_vintages.csv.gz")
```

## Current verified build

- History: 99,310 half-hour rows, from 1 January 2021 through 31 August 2026.
- History natural-key duplicates: zero.
- Maximum `dispatch_down - curtailment - constraint` difference: 0.031 MWh, below the 0.05 MWh
  rounding tolerance.
- Forecast archive: 287,389 normalized rows from 491 load vintages, 23 wind vintages, three daily
  interconnector-capacity vintages, and one solar snapshot.
- As-of feature table: 636 rows across 30- and 60-minute horizons.
- Forecast revisions selected after issue time: zero.

These figures are a snapshot of the 23 September 2026 build. The quality reports are the source of
truth after a later rerun.

## Time and leakage rules

EirGrid workbooks include a local timestamp and numeric GMT offset. UTC is calculated as local time
minus that offset, so the repeated autumn clock hour becomes two distinct UTC intervals. The original
local string and GMT offset remain in the normalized source frame for audit.

SEMO XML stores a target start/end time and an explicit report `PublishTime`. All revisions are kept.
For each model row the join matches the target interval, rejects publications after the issue time,
then selects the latest remaining publication. Duplicate copies of one revision are resolved by the
latest retrieval time.

The Smart Grid Dashboard supplies solar forecasts but no historical publication-vintage archive.
The pipeline therefore uses snapshot retrieval time as the publication/availability time. This is
conservative: a model can never use that snapshot before this project actually observed it.

## Coverage is allowed to be honest

The system schema grew over time. Solar, Greenlink, and several all-island fields do not exist in the
earliest workbook. The history table stays long and uses `<feature>_available_flag`; it does not drop
years to force complete cases and does not invent values. The quality report lists exact missing-cell
counts and source-level available columns.

The latest half-hourly labels are monthly publications and can lag the live forecast feed. The current
as-of table therefore reports `history_label_available_flag = 0` for its live targets. That table is
ready for inference and for future supervised training once the matching dispatch-down report is
published. Historical forecast coverage cannot be fabricated if the publisher no longer retains it.

The same gap means the current Issue #4 artifact cannot honestly calculate live forecast errors.
Completed system state is marked stale after 90 minutes, and stale interconnector flows are excluded
from headroom, utilisation and SNSP-proxy calculations. See
`docs/FORECAST_FEATURE_ENGINEERING.md` for formulas and availability rules.

## Source boundaries

- EirGrid system and dispatch-down workbooks come from the official System and Renewable Data Reports
  archive declared in `config/extended_data_sources.v1.json`.
- Wind, demand, and named-interconnector capacity vintages come from the official SEMO static-report
  API and retain the XML report publication time.
- Solar forecasts come from the official Smart Grid Dashboard variable-generation export and are
  accumulated as daily snapshots.

Raw publisher files are ignored by Git. Review the publisher's open-data licence and disclaimer before
redistributing them.
