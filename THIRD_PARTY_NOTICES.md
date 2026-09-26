# Third party notices

## Python packages

The data preparation notebook uses the versions recorded in `requirements.txt`.

- pandas - BSD 3-Clause License - tabular data processing
- NumPy - BSD 3-Clause License - numerical feature engineering
- openpyxl - MIT License - reading EirGrid Excel workbooks
- JupyterLab, IPython, ipykernel, nbformat, and nbclient - BSD-family licences - notebook execution
- scikit-learn - BSD 3-Clause License - classification, regression, and preprocessing pipelines
- joblib - BSD 3-Clause License - trusted model-bundle persistence
- FastAPI and Starlette - MIT/BSD-family licences - HTTP prediction API
- Uvicorn - BSD 3-Clause License - ASGI server
- HTTPX - BSD 3-Clause License - API integration testing

Package licence files distributed with the installed packages remain authoritative.

## Data sources

- Hack the Climate / ENTSO-E sample generation, load, and price files
- EirGrid System Data Qtr Hourly 2026
- EirGrid DD Half-Hourly 2026
- SEMO static reports for wind/load forecasts, RTIC schedules, physical notifications, forecast
  imbalance, net imbalance volume forecasts, interconnector NTC, imbalance outcomes, and SSII/SIFF

Source URLs and file hashes are recorded in `data/processed/source_manifest.csv`,
`data/processed/extended_source_manifest.csv`, and
`data/processed/semo_market_source_manifest.csv`. Data-provider terms and event-specific rules remain
authoritative; do not assume that public download access grants a right to redistribute unchanged raw
files. SEMO raw XML reports are downloaded from `https://reports.sem-o.com/` and are not redistributed
in this repository.
