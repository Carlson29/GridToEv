# Retrieve actual outcomes for either model

The prediction responses already contain the lookup key. Copy `target_timestamp_utc` from a v1 30/60-minute prediction, or `target_date_utc` from a daily-v2 prediction. The actuals endpoints read the EirGrid processed observation archive; they **do not** use model output as truth. If `GRIDTOEV_API_KEY` is configured, send the same `X-API-Key` header used for predictions.

| Prediction | Actuals endpoint | Actual fields |
| --- | --- | --- |
| v1 single, latest, or historical window item | `GET /actuals/v1?target_timestamp_utc=2026-01-15T12:30:00Z` | dispatch-down, curtailment and constraint MWh; dispatch-down event |
| v1 historical window array | `POST /actuals/v1/batch` with `{"target_timestamps_utc":["2026-01-15T12:30:00Z", "2026-01-15T13:00:00Z"]}` | same fields, array in request order (maximum 200) |
| Optional daily-v2 prediction | `GET /actuals/daily-curtailment?target_date_utc=2026-01-15` | complete UTC-day curtailment MWh and event |

`GET /actuals/coverage` gives the archive's earliest/latest half-hour and complete-day dates. All four routes are read-only and use the same optional API-key protection as the prediction routes. Timestamps must be timezone-aware ISO 8601 and align to a UTC half-hour. The daily date is `YYYY-MM-DD` in UTC, not Irish local time.

Every lookup returns `status`:

- `available`: the actual fields are populated. A genuine observed zero is `0.0`.
- `pending`: the target is later than the archive's latest observation, or its UTC day is still incomplete. Actual fields are `null`.
- `missing`: the target is within or before archive coverage but has no valid label or a complete-day label cannot be formed. Actual fields are `null`.

For v2, a day is `available` only when **all 48 half-hour EirGrid curtailment labels** are present and valid. For v1, the three MWh fields are returned only for a valid half-hour. No endpoint fills missing actuals with zero, and none calculates an error against an unavailable actual.

Example using PowerShell after obtaining a prediction:

```powershell
$base = "https://YOUR-SERVICE.onrender.com"
$headers = @{ "X-API-Key" = "YOUR-TEAM-KEY" } # Omit if no key is configured.
$prediction = Invoke-RestMethod -Uri "$base/predict/latest" -Headers $headers
$target = [uri]::EscapeDataString($prediction.predictions[0].target_timestamp_utc)
Invoke-RestMethod -Uri "$base/actuals/v1?target_timestamp_utc=$target" -Headers $headers
```

The bundled archive is a **snapshot**, currently ending at 2026-08-31 22:30 UTC, with complete daily labels through 2026-08-30. Later predictions will initially return `pending`. EirGrid actuals are published after the fact; to make a later outcome available, refresh `data/processed/eirgrid_core_history_30min.csv.gz` with the project's EirGrid history pipeline and redeploy the image. The service loads the archive lazily on first actuals lookup and does not auto-download or synthesize newer labels. `GRIDTOEV_ACTUALS_PATH` can select a different, verified archive; the Docker image includes the committed one.

These routes are proposed in the feature-branch PR only. They do not change v1's model artifact, release authorization, prediction schema, or existing endpoints.
