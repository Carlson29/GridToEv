# GridToEV project context

## Purpose

Prepare a reproducible, modelling-ready Irish renewable dispatch-down dataset for Hack the Climate
2026. The immediate deliverable is a Jupyter notebook and its processed CSV output.

## Stack

- Python 3.12
- pandas and NumPy for tabular processing and feature engineering
- openpyxl for EirGrid `.xlsx` inputs
- Jupyter for the documented workflow
- `unittest` for output regression checks

## Repository layout

- `notebooks/`: executable data preparation workflow
- `data/raw/`: downloaded source files, ignored by Git
- `data/processed/`: combined dataset, dictionary, provenance manifest, and quality report
- `tests/`: processed-output validation

## Verification

Run the notebook from the repository root, then run:

```powershell
python -m unittest discover -s tests -v
```

## Data constraints

Do not commit raw organiser or EirGrid downloads. Preserve UTC alignment, chronological evaluation,
and the separation between issue-time features and future targets.

