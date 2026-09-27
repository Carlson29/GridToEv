# Retrieve actual outcomes for either model

The prediction responses already contain the lookup key. Copy `target_timestamp_utc` from a v1 30/60-minute prediction, or `target_date_utc` from a daily-v2 prediction. The actuals endpoints read the EirGrid processed observation archive; they **do not** use model output as truth. If `GRIDTOEV_API_KEY` is configured, send the same `X-API-Key` header used for predictions.

| Prediction | Actuals endpoint | Actual fields |
| --- | --- | --- |
| v1 single, latest, or historical window item | `GET /actuals/v1?target_timestamp_utc=2026-01-15T12:30:00Z` | dispatch-down, curtailment and constraint MWh; dispatch-down event |
| v1 historical window array | `POST /actuals/v1/batch` with `{"target_timestamps_utc":["2026-01-15T12:30:00Z", "2026-01-15T13:00:00Z"]}` | same fields, array in request order (maximum 200) |
| v1 consecutive target half-hours | `POST /actuals/v1/window` with `{"start_target_timestamp_utc":"2026-01-15T12:30:00Z","duration_hours":2}` | ordered half-hour actuals and available/pending/missing counts (0.5–24.5 hours) |
| Optional daily-v2 prediction | `GET /actuals/daily-curtailment?target_date_utc=2026-01-15` | complete UTC-day curtailment MWh and event |
| v2 consecutive UTC days | `POST /actuals/daily-curtailment/window` with `{"start_date_utc":"2026-04-28","days":7}` | ordered daily actuals and available/pending/missing counts (1–7 days) |

For a V2 `/predict/curtailment/from-raw` result, copy **only** `target_date_utc` to `GET /actuals/daily-curtailment`; sending the 96 weather-forecast inputs again cannot produce an observed outcome. The new V2 actuals window accepts the same date-and-days body as `/predict/curtailment/window` and does not need the V2 model to be enabled. The V1 actuals window starts at the **first prediction's target timestamp**, not its issue timestamp. If a V1 prediction window spans 24 hours of issue times with both 30- and 60-minute horizons, request **24.5 hours** of target actuals to include the final 60-minute target. Match prediction and actual rows by `target_timestamp_utc`, not array position: the two horizons can predict the same target.

`GET /actuals/coverage` separates two different date ranges in one response. The top-level `available_target_timestamp_*` and `complete_day_*` fields describe the **EirGrid observed-outcome archive**. Its nested `v1_prediction_dataset.available_issue_timestamp_*` fields describe the loaded **V1 30/60-minute model-input dataset** instead. In the bundled snapshot, the actuals archive starts in 2021 and ends on 2026-08-31, whereas V1 issue times span 2026-01-02 through 2026-01-31. These are not supposed to match: actuals can exist on dates the V1 model-ready dataset cannot replay. The older top-level field names remain for existing clients. For full model-input coverage and accepted input format use `GET /dataset/info` (V1) or `GET /dataset/daily-curtailment/coverage` (V2). All six actuals routes are read-only and use the same optional API-key protection as the prediction routes. Timestamps must be timezone-aware ISO 8601 and align to a UTC half-hour. The daily date is `YYYY-MM-DD` in UTC, not Irish local time.

Archive minimum/maximum dates are bounds, not a promise that every intervening actual is valid; check the individual lookup's `status`.

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
$v1Window = @{start_target_timestamp_utc="2026-01-15T12:30:00Z"; duration_hours=2} | ConvertTo-Json
Invoke-RestMethod -Uri "$base/actuals/v1/window" -Method Post -ContentType "application/json" -Body $v1Window -Headers $headers
$v2Window = @{start_date_utc="2026-04-28"; days=7} | ConvertTo-Json
Invoke-RestMethod -Uri "$base/actuals/daily-curtailment/window" -Method Post -ContentType "application/json" -Body $v2Window -Headers $headers
```

The bundled archive is a **snapshot**, currently ending at 2026-08-31 22:30 UTC, with complete daily labels through 2026-08-30. Later predictions will initially return `pending`. EirGrid actuals are published after the fact; to make a later outcome available, refresh `data/processed/eirgrid_core_history_30min.csv.gz` with the project's EirGrid history pipeline and redeploy the image. The service loads the archive lazily on first actuals lookup and does not auto-download or synthesize newer labels. `GRIDTOEV_ACTUALS_PATH` can select a different, verified archive; the Docker image includes the committed one.

These routes do not change V1's model artifact or either model's prediction schema.
