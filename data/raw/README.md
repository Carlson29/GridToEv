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

Raw files are kept byte-for-byte and are excluded from Git. The processed `source_manifest.csv`
records the URL, size, and SHA-256 hash of every source used in a build. Review the relevant source
terms before redistributing raw data.

