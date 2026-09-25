# Issue #10: notebook, API, and release integration

## Current decision

The deployed prediction contract stays on the frozen **v1.1.0** model. Issue #8's best
point-forecast candidate improved rolling MAE by 1.22%, below the 15% release threshold,
and only two of four folds improved. Issue #9's experimental 10th–90th percentile interval
covered 68.91% of final-test rows, below the 78% minimum. Neither experiment produced an
approved, versioned replacement bundle. Do not present either experiment as the production
model or use its sealed-test score to tune the point model.

`benchmarks/release_preflight/release_report.json` records every gate, the candidate's
unavailable state, frozen rollback hashes, source coverage, feature order/hash, data split,
training settings, and library versions. Regenerate it after changing upstream evidence:

```powershell
python scripts/run_release_preflight.py
```

For a deployment gate, add `--require-candidate`; it returns a nonzero status while the
replacement remains unapproved, but still writes the diagnostic report. The default audit
command succeeds so the rejected evidence can be regenerated and inspected.

An alternate `GRIDTOEV_MODEL_PATH` will fail startup unless its SHA-256 matches a candidate
explicitly approved by the release report. The known v1.1.0 path is permitted only when its
bytes match the frozen contract checksum. `GRIDTOEV_RELEASE_REPORT_PATH` can select a report;
the committed report is the default. Changing a JSON flag alone must never be treated as a
release process: repeat the independent benchmarks, review the evidence, and only then issue
a versioned candidate artifact and signed-off release decision.

## Notebook and serving check

Run the two production notebooks end-to-end without touching the committed dataset or model:

```powershell
python scripts/verify_notebook_workflow.py --raw-cache data/raw
```

The check requires the three organiser CSVs and two EirGrid Excel files already cached under
`data/raw`. It copies them, the notebooks, and `src/` into a disposable directory, executes
notebook 01 then 02, checks their outputs, loads the newly trained bundle, and requests both
30- and 60-minute predictions. Its small evidence file is
`benchmarks/release_preflight/notebook_smoke_report.json`; the disposable model is deleted.
Run `python -m unittest discover -s tests -v` for API and release-gate tests.

The FastAPI flow is: the frontend asks `GET /model-info` for the model version, available
historical date range, release status, and ordered live-feature contract; then it sends a
complete feature snapshot to `POST /predict/features` or chooses a historical row through
`POST /predict/from-dataset`. Unknown, missing, non-finite, or incompatible features are
rejected with a clear 422 error. `/predict/latest` returns the two supported horizons. The
current interval and curtailment/constraint outputs still come from the frozen v1.1.0
bundle, not the experimental calibration from issue #9.

## Input-source decisions

| Source family | Current decision | Reason |
| --- | --- | --- |
| Original organiser/EirGrid features | Accepted | Present in the 119-feature frozen model contract and January labelled sample. |
| Forecast-error and grid-headroom | Unavailable for promotion | Engineering code exists, but there is no temporally overlapping labelled feature artifact for the frozen benchmark. |
| SEMO market/operational signals | Unavailable for promotion | The captured source period has zero overlap with the January benchmark. |
| Outage/constraint signals | Rejected for this release | Development ablation on its branch did not pass. |
| Regional weather signals | Rejected for this release | Development ablation on its branch did not pass. |
| Horizon ensembles and calibrated intervals | Experimental only | The point-model stability/improvement gates and interval-coverage gate failed. |

To finish issue #10's candidate-serving acceptance criterion, first obtain a genuinely
approved model through the frozen rolling-origin and sealed-test rules; ensure the exact
feature snapshot is available at prediction time; then version, checksum, and smoke-test
the bundle with this same API. Until then the deterministic rollback is the safe release.
