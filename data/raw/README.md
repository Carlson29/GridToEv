# Raw source data

The notebook creates the following folders and downloads files only when they are missing:

## Hack the Climate organiser samples

- `hacktheclimate/generation.csv` from <https://hacktheclimate.io/samples/generation.csv>
- `hacktheclimate/load.csv` from <https://hacktheclimate.io/samples/load.csv>
- `hacktheclimate/prices.csv` from <https://hacktheclimate.io/samples/prices.csv>

## EirGrid

- `eirgrid/system_data_qtr_hourly_2026_v7.xlsx` from
  <https://cms.eirgrid.ie/sites/default/files/publications/System-Data-Qtr-Hourly-2026-V7.xlsx>
- `eirgrid/dd_half_hourly_2026_v9.xlsx` from
  <https://cms.eirgrid.ie/sites/default/files/publications/DD-HH-2026-V9.xlsx>

The extended build stores immutable downloads by provider, report, and year:

- `eirgrid/system/<year-or-range>/`: official quarter-hourly system workbooks for 2020-2021,
  2022-2023, 2024, 2025, and the current 2026 publication. Processing begins at 2021.
- `eirgrid/dispatch_down/<year>/`: official half-hourly renewable dispatch-down workbooks for
  2021-present.
- `semo/forecast_vintages/<report>/<year>/<month>/`: individual SEMO XML publications for aggregated
  wind, daily load, and daily interconnector NTC forecasts.
- `eirgrid/forecast_snapshots/solar/<year>/`: daily Smart Grid Dashboard solar forecast snapshots.

Each newly downloaded file has a neighboring `.source.json` sidecar containing its original URL,
retrieval time, byte count, and SHA-256 checksum. `scripts/build_extended_data.py` skips a cached file
only after rechecking that checksum.

Raw files are kept byte-for-byte and are excluded from Git. The processed `source_manifest.csv`
records the URL, size, and SHA-256 hash of every source used in a build. Review the relevant source
terms before redistributing raw data.
