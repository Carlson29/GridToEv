# Shareable API: v2 release safeguards

This branch combines the historical prediction API with the purged v2 benchmark
contract. It does **not** deploy or promote a new model. The active bundle is
still the pinned v1.1.0 artifact.

At startup the API reads `benchmarks/release_preflight/release_report.v2.json`
and checks the model file's SHA-256 **before** loading it. The current report
verifies the rollback bundle and records that no replacement passed the v2
rolling, final-test, interval, and live-feature gates. An alternate artifact
path is rejected unless a future approved report names that exact path and
hash. Regenerate the report with:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python scripts/run_release_preflight.py
```

`--require-candidate` exits nonzero today by design. The report is evidence,
not an approval toggle; setting `candidate_approved` by hand does not create a
valid scientific evaluation.

`GET /model-info` now gives the ordered feature-list hash and required live
fields. `POST /predict/features` rejects missing, unknown, or non-finite
fields. Dataset/date guidance, the 48-hour historical replay limit, and
`X-API-Key` protection are unchanged. The API still cannot predict future
weather or grid conditions without a live feature source.

The default report path can be overridden with `GRIDTOEV_RELEASE_REPORT_PATH`.
The model path can be overridden with `GRIDTOEV_MODEL_PATH`, but a different
model will fail startup until its candidate release is genuinely approved.
Keep `GRIDTOEV_API_KEY` configured on a shared deployment.
