# Wind and solar curtailment: plain-English guide

## What is curtailment?

On windy or sunny days, Ireland's wind farms and solar farms can sometimes produce more electricity
than the grid can use or move. When that happens, the grid operator (EirGrid) tells some of them to
produce less. The clean electricity that *could* have been made but was not is called **curtailment**.
It is measured in **MWh (megawatt-hours)**. As a rough guide, 1 MWh could fully charge about 15-20
typical electric cars.

GridToEV's goal is to find those times, so that flexible demand such as EV charging can soak up
power that would otherwise be wasted.

## What these endpoints do

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

## Why is there no *forecast* of wind versus solar yet?

The API already forecasts **total** daily curtailment (the experimental V2 model). Splitting that
forecast into wind and solar was tested in two experiments:

1. **Standard machine-learning models** did no better than simply using last year's average
   monthly split, so they were rejected.
2. **A physics-based method** estimates each source's likely output from the weather forecast (wind
   speed and sunshine) scaled by how much wind and solar capacity is installed. EirGrid tends to cut
   sources in proportion to what they can produce, so this is a natural fit. It beat every
   alternative on 2025 and 2026 data. It is still **off** because the team had already looked at the
   2026 data while designing it, so it must also pass on newer days (at least 60 days from 31 August
   2026) before release.

The detailed numbers are in `benchmarks/daily_source_allocation_v2/` and in
`docs/SOURCE_CURTAILMENT_ATTRIBUTION_HANDOFF.md`. When the physics method passes, it will get its own
new forecast endpoint. These record endpoints will stay the same, so you can compare the forecast with
what really happened.

## Where the data comes from

EirGrid's public half-hourly "dispatch-down" workbooks for Ireland, 2021 onward. Each row gives a
technology (Wind or Solar) and how much of its output was curtailed. `scripts/build_source_curtailment_labels.py`
checks every file (no duplicates, no negative values, only Wind and Solar) and confirms that wind +
solar equals the existing total for every half-hour before the data is used. Review the EirGrid Open
Data Licence before redistributing the figures.
