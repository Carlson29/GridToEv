# Wind and solar curtailment: plain-English guide

This guide covers four endpoints: two that show **recorded** wind/solar curtailment for past days,
and two for an **experimental forecast** of the wind/solar split.

## What is curtailment?

On windy or sunny days, Ireland's wind farms and solar farms can sometimes produce more electricity
than the grid can use or move. When that happens, the grid operator (EirGrid) tells some of them to
produce less. The clean electricity that *could* have been made but was not is called **curtailment**.
It is measured in **MWh (megawatt-hours)**. As a rough guide, 1 MWh could fully charge about 15-20
typical electric cars.

GridToEV's goal is to find those times, so that flexible demand such as EV charging can soak up
power that would otherwise be wasted.

## Recorded wind/solar curtailment (past days)

Two new, read-only endpoints show **how much curtailment really happened on a past day, and how
much of it was wind versus solar**. The figures come from EirGrid's published half-hourly records.
They are **records, not forecasts**.

| Endpoint | In one sentence |
| --- | --- |
| `GET /actuals/curtailment/sources/coverage` | "Which days can I look up?" |
| `GET /actuals/curtailment/sources?target_date_utc=YYYY-MM-DD` | "On this day, how much wind and solar power was curtailed?" |

Both use the same API key as every other protected route (`X-API-Key` header). No existing endpoint
changed.

### 1. Which days can I look up?

`GET /actuals/curtailment/sources/coverage` takes no input. The important fields are:

- `complete_day_min_utc` / `complete_day_max_utc`: the first and last day with full wind **and**
  solar figures (currently 2023-04-01 to 2026-08-30).
- `solar_first_published_utc`: EirGrid started publishing Irish solar figures on 1 April 2023
  (shown in UTC as 2023-03-31T23:00). Before then, only wind figures exist.
- `last_half_hour_utc`: newer days are not available until the archive is refreshed.

### 2. How much was curtailed on a given day?

Send the day as `target_date_utc` in `YYYY-MM-DD` form. Days are **UTC** calendar days (in Irish
summer time, a UTC day runs from 01:00 to 01:00 local time).

Example request: `GET /actuals/curtailment/sources?target_date_utc=2026-05-10`

Example response (real data):

```json
{
  "status": "available",
  "target_date_utc": "2026-05-10",
  "source_latest_timestamp_utc": "2026-08-31T22:30:00+00:00",
  "wind_curtailment_mwh": 4352.908,
  "solar_curtailment_mwh": 2564.085,
  "total_curtailment_mwh": 6916.993,
  "wind_share_percent": 62.93,
  "solar_share_percent": 37.07,
  "complete_half_hour_count": 48,
  "summary": "On 2026-05-10 (UTC), 6,917.0 MWh of renewable power was curtailed in Ireland: 62.93% wind (4,352.9 MWh) and 37.07% solar (2,564.1 MWh)."
}
```

How to read it:

| Field | Meaning |
| --- | --- |
| `wind_curtailment_mwh` | Wind power wasted over the day. |
| `solar_curtailment_mwh` | Solar power wasted over the day. |
| `total_curtailment_mwh` | Wind + solar. It is the same number `/actuals/daily-curtailment` gives for that day. |
| `wind_share_percent` / `solar_share_percent` | Each one's share of the total. Empty (`null`) on days with no curtailment. |
| `summary` | The whole answer as one sentence, ready to show to a user. |

Add `&include_half_hours=true` to also get the 48 half-hour figures, for example to draw a chart of
when during the day the waste happened:

```json
{ "timestamp_utc": "2026-05-10T12:00:00+00:00", "wind_curtailment_mwh": 253.582, "solar_curtailment_mwh": 177.012 }
```

### What `status` means

| `status` | What it means | What you get |
| --- | --- | --- |
| `available` | Full wind and solar figures exist for all 48 half-hours. | All numbers. |
| `solar_not_published` | A day before April 2023. EirGrid only published wind figures then. | Wind MWh only. Solar is `null`, meaning **unknown**, not zero. |
| `pending` | EirGrid has not published this whole day yet (for example, a recent or future day). | No numbers. Try again after the archive is refreshed. |
| `missing` | The day is inside the archive, but its record is incomplete (for example, the day solar publishing started). | No numbers. |

The API never invents a zero. A `0.0` means EirGrid really recorded no curtailment; `null` means
the figure is not known.

## Try it

PowerShell:

```powershell
$base = "https://YOUR-SERVICE.onrender.com"
$headers = @{ "X-API-Key" = "YOUR-TEAM-KEY" }  # Omit if no key is configured.
Invoke-RestMethod -Uri "$base/actuals/curtailment/sources/coverage" -Headers $headers
Invoke-RestMethod -Uri "$base/actuals/curtailment/sources?target_date_utc=2026-05-10" -Headers $headers
```

Python:

```python
import requests

base = "https://YOUR-SERVICE.onrender.com"
headers = {"X-API-Key": "YOUR-TEAM-KEY"}
day = requests.get(f"{base}/actuals/curtailment/sources",
                   params={"target_date_utc": "2026-05-10"}, headers=headers, timeout=30).json()
print(day["summary"])
```

Other errors: `401` means the API key is missing or wrong; `422` means the date is not in
`YYYY-MM-DD` form; `503` means the server could not read its data file.

## Experimental forecast: wind vs solar for a day

The two routes above show what **happened**. These two **forecast** what will happen:

| Endpoint | In one sentence |
| --- | --- |
| `POST /predict/curtailment/sources/day` | "For this day, how much curtailment is expected, and how much of it from wind versus solar?" |
| `GET /model-info/curtailment/sources` | "How does the forecast work, which dates can I ask for, and how accurate is it so far?" |

They are labelled **experimental** (every response says `"experimental": true`). They need the V2
daily model to be switched on, which it is on the live service.

### How it works, in plain words

1. The V2 model predicts how much renewable power will be curtailed over the whole day.
2. EirGrid tends to cut wind and solar farms **in proportion to how much each could produce**.
   So the forecast looks at that day's weather forecast (wind speed and sunshine) and at how much
   wind and solar capacity is installed in Ireland.
3. It turns that into a percentage split and applies it to V2's total. Wind + solar always equals
   V2's total exactly.

### How to call it

Send the day you want as `target_date_utc` (`YYYY-MM-DD`, UTC). Any day from 2024-04-01 up to today
works. Nothing else is needed: the server fetches the archived day-ahead weather forecast itself.

```powershell
$body = @{ target_date_utc = "2026-09-15" } | ConvertTo-Json
Invoke-RestMethod -Uri "$base/predict/curtailment/sources/day" -Method Post -ContentType "application/json" -Body $body -Headers $headers
```

Example response (real, shortened):

```json
{
  "experimental": true,
  "validation_status": "candidate_awaiting_fresh_confirmation",
  "target_date_utc": "2026-09-15",
  "curtailment_event_probability": 0.9616,
  "predicted_curtailment_mwh": 5988.848,
  "predicted_wind_curtailment_mwh": 4927.074,
  "predicted_solar_curtailment_mwh": 1061.774,
  "predicted_wind_share_percent": 82.27,
  "predicted_solar_share_percent": 17.73,
  "capacity_proxy": { "solar_mw": 1321.745, "wind_mw": 4346.66, "published_data_through_utc": "2026-07-31" },
  "summary": "For 2026-09-15 (UTC), V2 predicts 5,988.8 MWh of curtailment. This experimental model expects about 82% of it from wind (4,927.1 MWh) and 18% from solar (1,061.8 MWh).",
  "compare_with_actual": "/actuals/curtailment/sources?target_date_utc=2026-09-15"
}
```

| Field | Meaning |
| --- | --- |
| `predicted_curtailment_mwh` | V2's forecast of the day's total wasted renewable power. Same as `/predict/curtailment/day`. |
| `predicted_wind_curtailment_mwh` / `predicted_solar_curtailment_mwh` | The expected wind and solar parts. |
| `predicted_wind_share_percent` / `predicted_solar_share_percent` | The same split as percentages. |
| `capacity_proxy` | The installed-capacity estimates used, and the last day of EirGrid data they came from. |
| `compare_with_actual` | After EirGrid publishes the day, call this to see what really happened. |
| `validation_status` | How far the unbiased check has got (see below). |

Errors: `422` for a future date, a date before 2024-04-01, a day with missing forecast hours, or a
date beyond the bundled capacity data (the message says what to refresh). `503` if V2 or the weather
provider is unavailable. `401` for a missing or wrong API key.

### Why you can trust that it does not cheat

- **No future information.** The weather inputs are forecasts made the day before. The capacity
  estimate only uses EirGrid months that were already published (always at least one full month
  before the target day). Actual curtailment is never an input.
- **Frozen model.** The forecast has only two fitted numbers, fitted once on 2024-2025 data and
  stored with a checksum. The server refuses to start the forecast if the file was changed.

### Why it is still called experimental

While designing this method, its designers had already seen 2026 results, so the 2026 accuracy
figures (about 2% better overall and 9% better on sunny curtailment days than the best simple
alternative) may be a little optimistic. The honest fix is to test it on **days that did not exist
yet** when it was designed. Every day from 31 August 2026 onward is such a day: its forecast is fixed
before its outcome exists, so nobody can bias it.

`GET /model-info/curtailment/sources` shows the current check under `validation.fresh_confirmation`.
Once EirGrid has published at least 60 such days (expected around late November 2026), rerun:

```powershell
python scripts/build_extended_data.py --forecast-days 3
python scripts/build_source_curtailment_labels.py
python scripts/build_daily_curtailment_data.py --end 2026-12-31 --output data/processed/daily_curtailment_confirmation_v2.csv.gz
python scripts/evaluate_daily_source_allocation.py --confirmation-dataset data/processed/daily_curtailment_confirmation_v2.csv.gz
```

Use a **separate** `--output` file for the extended daily dataset: the live V2 model checks the
checksum of its own frozen dataset and would refuse to start if it were overwritten. Set `--end` to the
last day EirGrid has published. These commands rebuild the data, but they **do not retrain** V2 or the split: both stay frozen, so the new days
remain a fair test. If the check passes, `validation_status` becomes `passed_release_gate` and the
experimental label can be removed. If it fails, switch the forecast off by setting
`GRIDTOEV_SOURCE_SPLIT_ENABLED=0`; nothing else is affected. Each served forecast is also written to the
server log (`source_split_prediction ...`), so what was actually served can be compared with the outcomes.

### Limits

- The split is only as good as V2's total, which can be off by thousands of MWh on a single day.
- It is a whole-day split; it does not say which hours were curtailed.
- The bundled capacity data ends on 2026-08-31, which covers target days up to **2026-10-31**. Later
  days return a clear 422 until the EirGrid archive is refreshed and redeployed.
- It is a planning estimate, not a dispatch instruction.

## Where the data comes from

EirGrid's public half-hourly "dispatch-down" workbooks for Ireland, 2021 onward. Each row gives a
technology (Wind or Solar) and how much of its output was curtailed. `scripts/build_source_curtailment_labels.py`
checks every file (no duplicates, no negative values, only Wind and Solar) and confirms that wind +
solar equals the existing total for every half-hour before the data is used. Review the EirGrid Open
Data Licence before redistributing the figures.
